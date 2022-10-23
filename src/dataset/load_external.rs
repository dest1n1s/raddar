use std::sync::Arc;

use super::{SimpleDataset, UnsupervisedDataset};
pub trait LoadFromJson {
    type ConfigType;

    fn from_json(path: &str, config: Self::ConfigType) -> Self;
}

pub struct SimpleDatasetJsonConfig {
    pub input_field: String,
    pub label_field: String,
}

impl<InputType, LabelType> LoadFromJson for SimpleDataset<InputType, LabelType>
where
    InputType: serde::de::DeserializeOwned,
    LabelType: serde::de::DeserializeOwned,
{
    type ConfigType = SimpleDatasetJsonConfig;

    fn from_json(path: &str, config: Self::ConfigType) -> Self {
        let file =
            std::fs::File::open(path).expect(format!("Failed to open file {}", path).as_str());
        let reader = std::io::BufReader::new(file);
        let mut inputs = Vec::new();
        let mut labels = Vec::new();
        let json: serde_json::Value =
            serde_json::from_reader(reader).expect("Failed to parse json");
        for item in json
            .as_array()
            .expect("Input file is not a valid JSON array")
        {
            let input = item
                .get(&config.input_field)
                .expect("Input field not found in JSON object");
            let label = item
                .get(&config.label_field)
                .expect("Label field not found in JSON object");
            inputs
                .push(Arc::new(serde_json::from_value(input.clone()).expect(
                    "Input field is not compatible with the specified type",
                )));
            labels
                .push(Arc::new(serde_json::from_value(label.clone()).expect(
                    "Label field is not compatible with the specified type",
                )));
        }
        Self::from_vectors(inputs, labels)
    }
}

pub struct UnsupervisedDatasetJsonConfig {
    pub input_field: String,
}

impl<InputType> LoadFromJson for UnsupervisedDataset<InputType>
where
    InputType: serde::de::DeserializeOwned,
{
    type ConfigType = UnsupervisedDatasetJsonConfig;

    fn from_json(path: &str, config: Self::ConfigType) -> Self {
        let file =
            std::fs::File::open(path).expect(format!("Failed to open file {}", path).as_str());
        let reader = std::io::BufReader::new(file);
        let mut inputs = Vec::new();
        let json: serde_json::Value =
            serde_json::from_reader(reader).expect("Failed to parse json");
        for item in json
            .as_array()
            .expect("Input file is not a valid JSON array")
        {
            let input = item
                .get(&config.input_field)
                .expect("Input field not found in JSON object");
            inputs
                .push(Arc::new(serde_json::from_value(input.clone()).expect(
                    "Input field is not compatible with the specified type",
                )));
        }
        Self::from_vectors(inputs)
    }
}

pub trait LoadFromImages {}
