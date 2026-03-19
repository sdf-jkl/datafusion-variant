use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray};
use arrow_schema::DataType;
use datafusion::common::{exec_datafusion_err, exec_err, plan_err};
use datafusion::error::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use datafusion::scalar::ScalarValue;
use parquet_variant::{Variant, VariantPath, VariantPathElement};
use parquet_variant_compute::VariantArray;

use crate::shared::{try_field_as_variant_array, try_parse_variant_scalar};

#[derive(Debug)]
enum VariantPathArgs {
    Array(ArrayRef),
    Scalars(Vec<Option<VariantPathElement<'static>>>),
}

impl VariantPathArgs {
    fn extract_path(path_args: &[ColumnarValue]) -> Result<Self> {
        if let Some((ColumnarValue::Array(array), rest)) = path_args.split_first()
            && rest.is_empty()
        {
            return Ok(Self::Array(Arc::clone(array)));
        }

        let mut parsed = Vec::with_capacity(path_args.len());
        for (idx, arg) in path_args.iter().enumerate() {
            match arg {
                ColumnarValue::Scalar(scalar) => {
                    parsed.push(parse_path_scalar(scalar, idx + 2)?);
                }
                ColumnarValue::Array(_) => {
                    return exec_err!(
                        "More than 1 path element is not supported when querying Variant using an array."
                    );
                }
            }
        }

        Ok(Self::Scalars(parsed))
    }
}

fn parse_path_scalar(
    scalar: &ScalarValue,
    arg_position: usize,
) -> Result<Option<VariantPathElement<'static>>> {
    match scalar {
        ScalarValue::Dictionary(_, value) => parse_path_scalar(value.as_ref(), arg_position),
        ScalarValue::Utf8(Some(value))
        | ScalarValue::Utf8View(Some(value))
        | ScalarValue::LargeUtf8(Some(value)) => Ok(Some(VariantPathElement::field(value.clone()))),
        ScalarValue::UInt64(Some(index)) => {
            Ok(usize::try_from(*index).ok().map(VariantPathElement::index))
        }
        ScalarValue::Int64(Some(index)) => {
            Ok(usize::try_from(*index).ok().map(VariantPathElement::index))
        }
        ScalarValue::Null
        | ScalarValue::Utf8(None)
        | ScalarValue::Utf8View(None)
        | ScalarValue::LargeUtf8(None)
        | ScalarValue::UInt64(None)
        | ScalarValue::Int64(None) => Ok(None),
        other => exec_err!(
            "Unexpected argument type at position {}, expected string or int, got {other:?}.",
            arg_position
        ),
    }
}

fn build_path(path: &[Option<VariantPathElement<'static>>]) -> Option<VariantPath<'static>> {
    let elements = path.iter().cloned().collect::<Option<Vec<_>>>()?;
    Some(VariantPath::new(elements))
}

fn variant_contains_path(variant: Option<Variant<'_, '_>>, path: Option<&VariantPath<'_>>) -> bool {
    let Some(variant) = variant else {
        return false;
    };
    let Some(path) = path else {
        return false;
    };

    variant.get_path(path).is_some()
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct VariantContainsUdf {
    signature: Signature,
}

impl Default for VariantContainsUdf {
    fn default() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for VariantContainsUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "variant_contains"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if arg_types.len() < 2 {
            plan_err!("The 'variant_contains' function requires two or more arguments.")
        } else {
            Ok(DataType::Boolean)
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let Some((variant_arg, path_args)) = args.args.split_first() else {
            return exec_err!("expected at least one argument");
        };

        if path_args.is_empty() {
            return exec_err!("expected at least one path argument");
        }

        let variant_field = args
            .arg_fields
            .first()
            .ok_or_else(|| exec_datafusion_err!("expected argument field"))?;
        try_field_as_variant_array(variant_field.as_ref())?;

        let path = VariantPathArgs::extract_path(path_args)?;

        let out = match (variant_arg, path) {
            (ColumnarValue::Array(variant_array), VariantPathArgs::Array(path_array)) => {
                if variant_array.len() != path_array.len() {
                    return exec_err!("expected variant array and path array to be of same length");
                }

                let variant_array = VariantArray::try_new(variant_array.as_ref())?;
                let values: Vec<bool> = (0..variant_array.len())
                    .map(|i| {
                        let path_scalar = ScalarValue::try_from_array(path_array.as_ref(), i)?;
                        let path = build_path(&[parse_path_scalar(&path_scalar, 2)?]);
                        Ok(variant_contains_path(
                            Some(variant_array.value(i)),
                            path.as_ref(),
                        ))
                    })
                    .collect::<Result<_>>()?;

                ColumnarValue::Array(Arc::new(BooleanArray::from(values)) as ArrayRef)
            }
            (ColumnarValue::Array(variant_array), VariantPathArgs::Scalars(path)) => {
                let path = build_path(&path);
                let variant_array = VariantArray::try_new(variant_array.as_ref())?;

                let values: Vec<bool> = variant_array
                    .iter()
                    .map(|v| variant_contains_path(v, path.as_ref()))
                    .collect();

                ColumnarValue::Array(Arc::new(BooleanArray::from(values)) as ArrayRef)
            }
            (ColumnarValue::Scalar(variant_scalar), VariantPathArgs::Array(path_array)) => {
                let variant_array = try_parse_variant_scalar(variant_scalar)?;
                let values: Vec<bool> = (0..path_array.len())
                    .map(|i| {
                        let path_scalar = ScalarValue::try_from_array(path_array.as_ref(), i)?;
                        let path = build_path(&[parse_path_scalar(&path_scalar, 2)?]);
                        Ok(variant_contains_path(
                            Some(variant_array.value(0)),
                            path.as_ref(),
                        ))
                    })
                    .collect::<Result<_>>()?;

                ColumnarValue::Array(Arc::new(BooleanArray::from(values)) as ArrayRef)
            }
            (ColumnarValue::Scalar(variant_scalar), VariantPathArgs::Scalars(path)) => {
                let path = build_path(&path);
                let variant_array = try_parse_variant_scalar(variant_scalar)?;
                let value = variant_contains_path(Some(variant_array.value(0)), path.as_ref());
                ColumnarValue::Scalar(ScalarValue::Boolean(Some(value)))
            }
        };

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, ArrayRef, BooleanArray, Int64Array, StringViewArray};
    use arrow_schema::{Field, Fields};
    use datafusion::logical_expr::ScalarFunctionArgs;
    use parquet_variant_compute::VariantType;

    use crate::shared::{variant_array_from_json_rows, variant_scalar_from_json};

    use super::*;

    fn make_args(variant_input: ColumnarValue, paths: Vec<ColumnarValue>) -> ScalarFunctionArgs {
        let mut args = vec![variant_input];
        args.extend(paths.iter().cloned());

        let mut arg_fields: Vec<Arc<Field>> = vec![Arc::new(
            Field::new("input", DataType::Struct(Fields::empty()), true)
                .with_extension_type(VariantType),
        )];
        for path in &paths {
            let data_type = match path {
                ColumnarValue::Array(array) => array.data_type().clone(),
                ColumnarValue::Scalar(scalar) => scalar.data_type(),
            };
            arg_fields.push(Arc::new(Field::new("path", data_type, true)));
        }

        ScalarFunctionArgs {
            args,
            return_field: Arc::new(Field::new("result", DataType::Boolean, true)),
            arg_fields,
            number_rows: Default::default(),
            config_options: Default::default(),
        }
    }

    #[test]
    fn test_scalar_single_key() {
        let udf = VariantContainsUdf::default();
        let args = make_args(
            ColumnarValue::Scalar(variant_scalar_from_json(serde_json::json!({"a": 1}))),
            vec![ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                "a".to_string(),
            )))],
        );

        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Boolean(Some(value))) = result else {
            panic!("expected bool scalar");
        };
        assert!(value);
    }

    #[test]
    fn test_scalar_nested_path() {
        let udf = VariantContainsUdf::default();
        let args = make_args(
            ColumnarValue::Scalar(variant_scalar_from_json(serde_json::json!({"a": {"b": 1}}))),
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("a".to_string()))),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("b".to_string()))),
            ],
        );

        let result = udf.invoke_with_args(args).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Boolean(Some(value))) = result else {
            panic!("expected bool scalar");
        };
        assert!(value);
    }

    #[test]
    fn test_dot_notation_treated_as_single_key() {
        let udf = VariantContainsUdf::default();
        let variant = ColumnarValue::Scalar(variant_scalar_from_json(serde_json::json!({
            "a.b": 1
        })));

        let dotted_key = make_args(
            variant.clone(),
            vec![ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                "a.b".to_string(),
            )))],
        );
        let result = udf.invoke_with_args(dotted_key).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Boolean(Some(value))) = result else {
            panic!("expected bool scalar");
        };
        assert!(value);

        let split_key = make_args(
            variant,
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("a".to_string()))),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("b".to_string()))),
            ],
        );
        let result = udf.invoke_with_args(split_key).unwrap();
        let ColumnarValue::Scalar(ScalarValue::Boolean(Some(value))) = result else {
            panic!("expected bool scalar");
        };
        assert!(!value);
    }

    #[test]
    fn test_array_input_scalar_path() {
        let udf = VariantContainsUdf::default();
        let variant_array = variant_array_from_json_rows(&[
            serde_json::json!({"a": 1}),
            serde_json::json!({"b": 2}),
            serde_json::json!(null),
        ]);

        let args = make_args(
            ColumnarValue::Array(variant_array),
            vec![ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                "a".to_string(),
            )))],
        );
        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(array) = result else {
            panic!("expected bool array");
        };
        let values = array.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(values.len(), 3);
        assert_eq!(values.value(0), true);
        assert_eq!(values.value(1), false);
        assert_eq!(values.value(2), false);
    }

    #[test]
    fn test_scalar_input_array_paths() {
        let udf = VariantContainsUdf::default();
        let path_array: ArrayRef = Arc::new(Int64Array::from(vec![Some(0), Some(1), None]));
        let args = make_args(
            ColumnarValue::Scalar(variant_scalar_from_json(serde_json::json!([10, 20]))),
            vec![ColumnarValue::Array(path_array)],
        );
        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(array) = result else {
            panic!("expected bool array");
        };
        let values = array.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(values.len(), 3);
        assert_eq!(values.value(0), true);
        assert_eq!(values.value(1), true);
        assert_eq!(values.value(2), false);
    }

    #[test]
    fn test_array_input_array_paths() {
        let udf = VariantContainsUdf::default();
        let variant_array = variant_array_from_json_rows(&[
            serde_json::json!({"a": 1}),
            serde_json::json!({"b": 2}),
            serde_json::json!({"a.b": 3}),
        ]);
        let path_array: ArrayRef = Arc::new(StringViewArray::from(vec!["a", "b", "a.b"]));

        let args = make_args(
            ColumnarValue::Array(variant_array),
            vec![ColumnarValue::Array(path_array)],
        );
        let result = udf.invoke_with_args(args).unwrap();

        let ColumnarValue::Array(array) = result else {
            panic!("expected bool array");
        };
        let values = array.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(values.len(), 3);
        assert_eq!(values.value(0), true);
        assert_eq!(values.value(1), true);
        assert_eq!(values.value(2), true);
    }
}
