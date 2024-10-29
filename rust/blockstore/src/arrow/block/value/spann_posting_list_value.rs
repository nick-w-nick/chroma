use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, ListArray, StructArray, UInt32Array};
use chroma_types::SpannPostingList;

use crate::{
    arrow::{
        block::delta::{spann_posting_list_delta::SpannPostingListDelta, BlockDelta, BlockStorage},
        types::{ArrowReadableValue, ArrowWriteableKey, ArrowWriteableValue},
    },
    key::KeyWrapper,
};

impl ArrowWriteableValue for &SpannPostingList<'_> {
    type ReadableValue<'referred_data> = SpannPostingList<'referred_data>;

    // This method is only called for SingleColumnStorage.
    fn offset_size(_: usize) -> usize {
        unimplemented!()
    }

    // This method is only called for SingleColumnStorage.
    fn validity_size(_: usize) -> usize {
        unimplemented!()
    }

    fn add(prefix: &str, key: KeyWrapper, value: Self, delta: &BlockDelta) {
        match &delta.builder {
            BlockStorage::SpannPostingListDelta(builder) => {
                builder.add(prefix, key, value);
            }
            _ => panic!("Invalid builder type"),
        }
    }

    fn delete(prefix: &str, key: KeyWrapper, delta: &BlockDelta) {
        match &delta.builder {
            BlockStorage::SpannPostingListDelta(builder) => {
                builder.delete(prefix, key);
            }
            _ => panic!("Invalid builder type"),
        }
    }

    fn get_delta_builder() -> BlockStorage {
        BlockStorage::SpannPostingListDelta(SpannPostingListDelta::new())
    }
}

impl<'referred_data> ArrowReadableValue<'referred_data> for SpannPostingList<'referred_data> {
    fn get(array: &'referred_data Arc<dyn Array>, index: usize) -> Self {
        let as_struct_array = array.as_any().downcast_ref::<StructArray>().unwrap();

        let doc_offset_ids_arr = as_struct_array
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let doc_id_start_idx = doc_offset_ids_arr.value_offsets()[index] as usize;
        let doc_id_end_idx = doc_offset_ids_arr.value_offsets()[index + 1] as usize;

        let doc_offset_slice_at_idx = &doc_offset_ids_arr
            .values()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .values()[doc_id_start_idx..doc_id_end_idx];

        let doc_versions_arr = as_struct_array
            .column(1)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let doc_version_start_idx = doc_versions_arr.value_offsets()[index] as usize;
        let doc_version_end_idx = doc_versions_arr.value_offsets()[index + 1] as usize;
        let doc_versions_slice_at_idx = &doc_versions_arr
            .values()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap()
            .values()[doc_version_start_idx..doc_version_end_idx];

        let doc_embeddings_arr = as_struct_array
            .column(2)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let top_level_start_idx = doc_embeddings_arr.value_offsets()[index] as usize;
        let top_level_end_idx = doc_embeddings_arr.value_offsets()[index + 1] as usize;
        let doc_embeddings_fixed_size_list = doc_embeddings_arr
            .values()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        let doc_embeddings_start_idx =
            doc_embeddings_fixed_size_list.value_offset(top_level_start_idx) as usize;
        let doc_embeddings_end_idx =
            doc_embeddings_fixed_size_list.value_offset(top_level_end_idx) as usize;
        let doc_embeddings_slice_at_idx = &doc_embeddings_fixed_size_list
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()[doc_embeddings_start_idx..doc_embeddings_end_idx];

        SpannPostingList {
            doc_offset_ids: doc_offset_slice_at_idx,
            doc_versions: doc_versions_slice_at_idx,
            doc_embeddings: doc_embeddings_slice_at_idx,
        }
    }

    fn add_to_delta<K: ArrowWriteableKey>(
        prefix: &str,
        key: K,
        value: Self,
        delta: &mut BlockDelta,
    ) {
        delta.add(prefix, key, &value);
    }
}
