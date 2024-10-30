use std::{collections::BTreeMap, sync::Arc};

use arrow::{
    array::{
        Array, ArrayRef, FixedSizeListBuilder, Float32Builder, ListBuilder, RecordBatch,
        StructArray, UInt32Builder,
    },
    datatypes::{DataType, Field, Fields},
    util::bit_util,
};
use chroma_types::SpannPostingList;
use parking_lot::RwLock;

use crate::{
    arrow::types::ArrowWriteableKey,
    key::{CompositeKey, KeyWrapper},
};

use super::BlockKeyArrowBuilder;

pub type SpannPostingListDeltaEntry = (Vec<u32>, Vec<u32>, Vec<f32>);

#[derive(Debug)]
struct Inner {
    storage: BTreeMap<CompositeKey, SpannPostingListDeltaEntry>,
    prefix_size: usize,
    key_size: usize,
    doc_offset_ids_size: usize,
    doc_versions_size: usize,
    doc_embeddings_size: usize,
}

struct SplitInformation {
    split_key: CompositeKey,
    remaining_prefix_size: usize,
    remaining_key_size: usize,
    remaining_doc_offset_ids_size: usize,
    remaining_doc_versions_size: usize,
    remaining_doc_embeddings_size: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct SpannPostingListDelta {
    inner: Arc<RwLock<Inner>>,
}

impl SpannPostingListDelta {
    pub(in crate::arrow) fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Inner {
                storage: BTreeMap::new(),
                prefix_size: 0,
                key_size: 0,
                doc_offset_ids_size: 0,
                doc_versions_size: 0,
                doc_embeddings_size: 0,
            })),
        }
    }

    pub(super) fn get_prefix_size(&self) -> usize {
        self.inner.read().prefix_size
    }

    pub(super) fn get_key_size(&self) -> usize {
        self.inner.read().key_size
    }

    pub fn get_owned_value(
        &self,
        prefix: &str,
        key: KeyWrapper,
    ) -> Option<SpannPostingListDeltaEntry> {
        let read_guard = self.inner.read();
        let composite_key = CompositeKey {
            prefix: prefix.to_string(),
            key,
        };
        read_guard.storage.get(&composite_key).cloned()
    }

    pub fn add(&self, prefix: &str, key: KeyWrapper, value: &SpannPostingList<'_>) {
        let mut lock_guard = self.inner.write();
        let composite_key = CompositeKey {
            prefix: prefix.to_string(),
            key,
        };
        // Subtract the old sizes. Remove the old posting list if it exists.
        if let Some(pl) = lock_guard.storage.remove(&composite_key) {
            lock_guard.doc_offset_ids_size -= pl.0.len() * std::mem::size_of::<u32>();
            lock_guard.doc_versions_size -= pl.1.len() * std::mem::size_of::<u32>();
            lock_guard.doc_embeddings_size -= pl.2.len() * std::mem::size_of::<f32>();
            lock_guard.prefix_size -= composite_key.prefix.len();
            lock_guard.key_size -= composite_key.key.get_size();
        }
        // Add the new sizes.
        lock_guard.prefix_size += composite_key.prefix.len();
        lock_guard.key_size += composite_key.key.get_size();
        lock_guard.doc_offset_ids_size += std::mem::size_of_val(value.doc_offset_ids);
        lock_guard.doc_versions_size += std::mem::size_of_val(value.doc_versions);
        lock_guard.doc_embeddings_size += std::mem::size_of_val(value.doc_embeddings);
        // Add the value in the btree.
        lock_guard.storage.insert(
            composite_key,
            (
                value.doc_offset_ids.to_vec(),
                value.doc_versions.to_vec(),
                value.doc_embeddings.to_vec(),
            ),
        );
    }

    pub fn delete(&self, prefix: &str, key: KeyWrapper) {
        let mut lock_guard = self.inner.write();
        let composite_key = CompositeKey {
            prefix: prefix.to_string(),
            key,
        };
        if let Some(pl) = lock_guard.storage.remove(&composite_key) {
            lock_guard.doc_offset_ids_size -= pl.0.len() * std::mem::size_of::<u32>();
            lock_guard.doc_versions_size -= pl.1.len() * std::mem::size_of::<u32>();
            lock_guard.doc_embeddings_size -= pl.2.len() * std::mem::size_of::<f32>();
            lock_guard.prefix_size -= composite_key.prefix.len();
            lock_guard.key_size -= composite_key.key.get_size();
        }
    }

    pub(super) fn get_size<K: ArrowWriteableKey>(&self) -> usize {
        let read_guard = self.inner.read();
        let prefix_size = bit_util::round_upto_multiple_of_64(read_guard.prefix_size);
        let key_size = bit_util::round_upto_multiple_of_64(read_guard.key_size);
        let doc_offset_ids_size =
            bit_util::round_upto_multiple_of_64(read_guard.doc_offset_ids_size);
        let doc_versions_size = bit_util::round_upto_multiple_of_64(read_guard.doc_versions_size);
        let doc_embeddings_size =
            bit_util::round_upto_multiple_of_64(read_guard.doc_embeddings_size);

        // Account for offsets.
        let num_elts = read_guard.storage.len();
        let prefix_offset_size =
            bit_util::round_upto_multiple_of_64((num_elts + 1) * std::mem::size_of::<i32>());
        let key_offset_size = K::offset_size(num_elts);
        let doc_offset_ids_offset_size =
            bit_util::round_upto_multiple_of_64((num_elts + 1) * std::mem::size_of::<i32>());
        let doc_versions_offset_size =
            bit_util::round_upto_multiple_of_64((num_elts + 1) * std::mem::size_of::<i32>());
        // validity bitmap for fixed size embeddings list not required since it is not null.
        let doc_embeddings_offset_size =
            bit_util::round_upto_multiple_of_64((num_elts + 1) * std::mem::size_of::<i32>());
        prefix_size
            + key_size
            + doc_offset_ids_size
            + doc_versions_size
            + doc_embeddings_size
            + prefix_offset_size
            + key_offset_size
            + doc_offset_ids_offset_size
            + doc_versions_offset_size
            + doc_embeddings_offset_size
    }

    // assumes there is a split point.
    fn split_internal<K: ArrowWriteableKey>(&self, split_size: usize) -> SplitInformation {
        let mut cumulative_prefix_size = 0;
        let mut cumulative_key_size = 0;
        let mut cumulative_offset_ids_size = 0;
        let mut cumulative_versions_size = 0;
        let mut cumulative_embeddings_size = 0;
        let mut cumulative_count = 0;
        let mut split_key = None;

        let read_guard = self.inner.read();
        for (key, pl) in &read_guard.storage {
            cumulative_count += 1;
            cumulative_prefix_size += key.prefix.len();
            cumulative_key_size += key.key.get_size();
            cumulative_offset_ids_size += pl.0.len() * std::mem::size_of::<u32>();
            cumulative_versions_size += pl.1.len() * std::mem::size_of::<u32>();
            cumulative_embeddings_size += pl.2.len() * std::mem::size_of::<f32>();

            let prefix_offset_size = bit_util::round_upto_multiple_of_64(
                (cumulative_count + 1) * std::mem::size_of::<i32>(),
            );
            let key_offset_size = K::offset_size(cumulative_count);
            let doc_offset_ids_offset_size = bit_util::round_upto_multiple_of_64(
                (cumulative_count + 1) * std::mem::size_of::<i32>(),
            );
            let doc_versions_offset_size = bit_util::round_upto_multiple_of_64(
                (cumulative_count + 1) * std::mem::size_of::<i32>(),
            );
            let doc_embeddings_offset_size = bit_util::round_upto_multiple_of_64(
                (cumulative_count + 1) * std::mem::size_of::<i32>(),
            );
            let total_size = bit_util::round_upto_multiple_of_64(cumulative_prefix_size)
                + bit_util::round_upto_multiple_of_64(cumulative_key_size)
                + bit_util::round_upto_multiple_of_64(cumulative_offset_ids_size)
                + bit_util::round_upto_multiple_of_64(cumulative_versions_size)
                + bit_util::round_upto_multiple_of_64(cumulative_embeddings_size)
                + prefix_offset_size
                + key_offset_size
                + doc_offset_ids_offset_size
                + doc_versions_offset_size
                + doc_embeddings_offset_size;

            if total_size > split_size {
                split_key = Some(key.clone());
                cumulative_prefix_size -= key.prefix.len();
                cumulative_key_size -= key.key.get_size();
                cumulative_offset_ids_size -= pl.0.len() * std::mem::size_of::<u32>();
                cumulative_versions_size -= pl.1.len() * std::mem::size_of::<u32>();
                cumulative_embeddings_size -= pl.2.len() * std::mem::size_of::<f32>();
                break;
            }
        }
        SplitInformation {
            split_key: split_key.expect("Split key expected to be found"),
            remaining_prefix_size: read_guard.prefix_size - cumulative_prefix_size,
            remaining_key_size: read_guard.key_size - cumulative_key_size,
            remaining_doc_offset_ids_size: read_guard.doc_offset_ids_size
                - cumulative_offset_ids_size,
            remaining_doc_versions_size: read_guard.doc_versions_size - cumulative_versions_size,
            remaining_doc_embeddings_size: read_guard.doc_embeddings_size
                - cumulative_embeddings_size,
        }
    }

    pub(super) fn split<K: ArrowWriteableKey>(
        &self,
        split_size: usize,
    ) -> (CompositeKey, SpannPostingListDelta) {
        let split_info = self.split_internal::<K>(split_size);
        let mut write_guard = self.inner.write();
        write_guard.prefix_size -= split_info.remaining_prefix_size;
        write_guard.key_size -= split_info.remaining_key_size;
        write_guard.doc_offset_ids_size -= split_info.remaining_doc_offset_ids_size;
        write_guard.doc_versions_size -= split_info.remaining_doc_versions_size;
        write_guard.doc_embeddings_size -= split_info.remaining_doc_embeddings_size;
        let new_storage = write_guard.storage.split_off(&split_info.split_key);
        (
            split_info.split_key,
            SpannPostingListDelta {
                inner: Arc::new(RwLock::new(Inner {
                    storage: new_storage,
                    prefix_size: split_info.remaining_prefix_size,
                    key_size: split_info.remaining_key_size,
                    doc_offset_ids_size: split_info.remaining_doc_offset_ids_size,
                    doc_versions_size: split_info.remaining_doc_versions_size,
                    doc_embeddings_size: split_info.remaining_doc_embeddings_size,
                })),
            },
        )
    }

    pub fn get_min_key(&self) -> Option<CompositeKey> {
        self.inner.read().storage.keys().next().cloned()
    }

    pub(super) fn len(&self) -> usize {
        self.inner.read().storage.len()
    }

    pub(super) fn into_arrow(
        self,
        key_builder: BlockKeyArrowBuilder,
    ) -> Result<RecordBatch, arrow::error::ArrowError> {
        // build arrow key.
        let mut key_builder = key_builder;
        let mut offset_struct_builder;
        let mut version_struct_builder;
        let mut embeddings_struct_builder;
        let mut embedding_dim = 0;
        match Arc::try_unwrap(self.inner) {
            Ok(inner) => {
                let inner = inner.into_inner();
                let storage = inner.storage;
                let num_heads = storage.len();
                if num_heads == 0 {
                    // ok to initialize embedding dim to 0.
                    embedding_dim = 0;
                    offset_struct_builder = ListBuilder::new(UInt32Builder::new());
                    version_struct_builder = ListBuilder::new(UInt32Builder::new());
                    embeddings_struct_builder =
                        ListBuilder::new(FixedSizeListBuilder::new(Float32Builder::new(), 0));
                } else {
                    // Compute the embedding dimension needed for builders preallocation.
                    // Assumes all embeddings are of the same length, which is guaranteed by calling code
                    // TODO: validate this assumption by throwing an error if it's not true
                    let mut num_postings = 0;
                    let mut num_embeddings = 0;
                    storage.iter().for_each(|(_, pl)| {
                        if !pl.0.is_empty() {
                            embedding_dim = pl.2.len() / pl.0.len();
                            num_postings += pl.0.len();
                            num_embeddings += pl.2.len();
                        }
                    });
                    // Construct the builders.
                    offset_struct_builder = ListBuilder::with_capacity(
                        UInt32Builder::with_capacity(num_postings),
                        num_heads,
                    );
                    version_struct_builder = ListBuilder::with_capacity(
                        UInt32Builder::with_capacity(num_postings),
                        num_heads,
                    );
                    embeddings_struct_builder = ListBuilder::with_capacity(
                        FixedSizeListBuilder::with_capacity(
                            Float32Builder::with_capacity(num_embeddings),
                            embedding_dim as i32,
                            num_postings,
                        ),
                        num_heads,
                    );
                }
                // TODO: Add null if posting list is empty.
                for (key, (doc_offset_ids, doc_versions, doc_embeddings)) in storage.into_iter() {
                    key_builder.add_key(key);
                    let inner_offset_id_ref = offset_struct_builder.values();
                    let inner_version_ref = version_struct_builder.values();
                    for (doc_offset_id, doc_version) in
                        doc_offset_ids.into_iter().zip(doc_versions.into_iter())
                    {
                        inner_offset_id_ref.append_value(doc_offset_id);
                        inner_version_ref.append_value(doc_version);
                    }
                    let inner_embeddings_ref = embeddings_struct_builder.values();
                    let mut f32_count = 0;
                    for embedding in doc_embeddings.into_iter() {
                        inner_embeddings_ref.values().append_value(embedding);
                        f32_count += 1;
                        if f32_count == embedding_dim {
                            inner_embeddings_ref.append(true);
                            f32_count = 0;
                        }
                    }
                    offset_struct_builder.append(true);
                    version_struct_builder.append(true);
                    embeddings_struct_builder.append(true);
                }
            }
            Err(_) => {
                panic!("Invariant violation: Spann posting list delta inner should have only one reference.");
            }
        }
        // Build arrow key with fields.
        let (prefix_field, prefix_arr, key_field, key_arr) = key_builder.as_arrow();

        // Struct fields.
        let offset_field = Field::new(
            "offset_ids",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
            true,
        );
        let version_field = Field::new(
            "version",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
            true,
        );
        let embeddings_field = Field::new(
            "embeddings",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    embedding_dim as i32,
                ),
                true,
            ))),
            true,
        );
        // Construct struct array from these 3 child arrays.
        let offset_child_array = offset_struct_builder.finish();
        let version_child_array = version_struct_builder.finish();
        let embeddings_child_array = embeddings_struct_builder.finish();
        let value_arr = StructArray::from(vec![
            (
                Arc::new(offset_field.clone()),
                Arc::new(offset_child_array) as ArrayRef,
            ),
            (
                Arc::new(version_field.clone()),
                Arc::new(version_child_array) as ArrayRef,
            ),
            (
                Arc::new(embeddings_field.clone()),
                Arc::new(embeddings_child_array) as ArrayRef,
            ),
        ]);
        let struct_fields = Fields::from(vec![offset_field, version_field, embeddings_field]);
        let value_field = Field::new("value", DataType::Struct(struct_fields), true);
        let value_arr = (&value_arr as &dyn Array).slice(0, value_arr.len());

        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            prefix_field,
            key_field,
            value_field,
        ]));
        RecordBatch::try_new(schema, vec![prefix_arr, key_arr, value_arr])
    }
}
