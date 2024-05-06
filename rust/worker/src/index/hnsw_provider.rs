use super::{HnswIndex, HnswIndexConfig, Index, IndexConfig, IndexConfigFromSegmentError};
use crate::errors::ErrorCodes;
use crate::index::types::PersistentIndex;
use crate::{errors::ChromaError, storage::Storage, types::Segment};
use parking_lot::RwLock;
use std::fmt::Debug;
use std::path::Path;
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use thiserror::Error;
use uuid::Uuid;

// These are the files hnswlib writes to disk. This is strong coupling, but we need to know
// what files to read from disk. We could in the future have the C++ code return the files
// but ideally we have a rust implementation of hnswlib
const FILES: [&'static str; 4] = [
    "header.bin",
    "data_level0.bin",
    "length.bin",
    "link_lists.bin",
];

#[derive(Clone)]
pub(crate) struct HnswIndexProvider {
    cache: Arc<RwLock<HashMap<Uuid, Arc<RwLock<HnswIndex>>>>>,
    pub(crate) temporary_storage_path: PathBuf,
    storage: Box<Storage>,
}

impl Debug for HnswIndexProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HnswIndexProvider {{ temporary_storage_path: {:?}, cache: {} }}",
            self.temporary_storage_path,
            self.cache.read().len(),
        )
    }
}

impl HnswIndexProvider {
    pub(crate) fn new(storage: Box<Storage>, storage_path: PathBuf) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            storage,
            temporary_storage_path: storage_path,
        }
    }

    pub(crate) fn get(&self, id: &Uuid) -> Option<Arc<RwLock<HnswIndex>>> {
        let cache = self.cache.read();
        cache.get(id).cloned()
    }

    fn format_key(&self, id: &Uuid, file: &str) -> String {
        format!("hnsw/{}/{}", id, file)
    }

    pub(crate) async fn fork(
        &self,
        source_id: &Uuid,
        segment: &Segment,
        dimensionality: i32,
    ) -> Result<Arc<RwLock<HnswIndex>>, Box<dyn ChromaError>> {
        let new_id = Uuid::new_v4();
        let new_storage_path = self.temporary_storage_path.join(new_id.to_string());
        self.create_dir_all(&new_storage_path)?;
        self.load_hnsw_segment_into_directory(source_id, &new_storage_path)
            .await?;

        let index_config = IndexConfig::from_segment(&segment, dimensionality);

        let index_config = match index_config {
            Ok(index_config) => index_config,
            Err(e) => {
                return Err(e);
            }
        };

        let hnsw_config = HnswIndexConfig::from_segment(segment, &new_storage_path)?;
        // TODO: don't unwrap path conv here
        match HnswIndex::load(
            new_storage_path.to_str().unwrap(),
            &index_config,
            *source_id,
        ) {
            Ok(index) => {
                let index = Arc::new(RwLock::new(index));
                let mut cache = self.cache.write();
                cache.insert(new_id, index.clone());
                Ok(index)
            }
            Err(e) => Err(e),
        }
    }

    async fn load_hnsw_segment_into_directory(
        &self,
        source_id: &Uuid,
        index_storage_path: &Path,
    ) -> Result<(), Box<dyn ChromaError>> {
        // Fetch the files from storage and put them in the index storage path
        for file in FILES.iter() {
            let key = self.format_key(source_id, file);
            println!("Loading hnsw index file: {}", key);
            let res = self.storage.get(&key).await;
            let mut reader = match res {
                Ok(reader) => reader,
                Err(e) => {
                    // TODO: return Err(e);
                    panic!("Failed to load hnsw index file from storage: {}", e);
                }
            };

            let file_path = index_storage_path.join(file);
            // For now, we never evict from the cache, so if the index is being loaded, the file does not exist
            let file_handle = tokio::fs::File::create(&file_path).await;
            let mut file_handle = match file_handle {
                Ok(file) => file,
                Err(e) => {
                    // TODO: cleanup created files if this fails
                    panic!("Failed to create file: {}", e);
                }
            };
            let copy_res = tokio::io::copy(&mut reader, &mut file_handle).await;
            match copy_res {
                Ok(_) => {
                    println!(
                        "Copied storage key: {} to file: {}",
                        key,
                        file_path.to_str().unwrap()
                    );
                }
                Err(e) => {
                    // TODO: cleanup created files if this fails and error handle
                    panic!("Failed to copy file: {}", e);
                }
            }
            // bytes is an AsyncBufRead, so we fil and consume it to a file
            println!("Loaded hnsw index file: {}", file);
        }
        Ok(())
    }

    pub(crate) async fn open(
        &self,
        id: &Uuid,
        segment: &Segment,
        dimensionality: i32,
    ) -> Result<Arc<RwLock<HnswIndex>>, Box<HnswIndexProviderOpenError>> {
        let index_storage_path = self.temporary_storage_path.join(id.to_string());

        match self.create_dir_all(&index_storage_path) {
            Ok(_) => {}
            Err(e) => {
                return Err(Box::new(HnswIndexProviderOpenError::FileError(e)));
            }
        }

        self.load_hnsw_segment_into_directory(id, &index_storage_path)
            .await?;

        let index_config = IndexConfig::from_segment(&segment, dimensionality)?;
        let index_config = match index_config {
            Ok(index_config) => index_config,
            Err(e) => {
                return Err(e);
            }
        };

        let hnsw_config = HnswIndexConfig::from_segment(segment, &index_storage_path)?;
        // TODO: don't unwrap path conv here
        match HnswIndex::load(index_storage_path.to_str().unwrap(), &index_config, *id) {
            Ok(index) => {
                let index = Arc::new(RwLock::new(index));
                let mut cache = self.cache.write();
                cache.insert(*id, index.clone());
                Ok(index)
            }
            Err(e) => Err(e),
        }
    }

    // Compactor
    // Cases
    // A write comes in and no files are in the segment -> we know we need to create a new index
    // A write comes in and files are in the segment -> we know we need to load the index
    // If the writer drops, but we already have the index, the id will be in the cache and the next job will have files and not need to load the index

    // Query
    // Cases
    // A query comes in and the index is in the cache -> we can query the index based on segment files id (Same as compactor case 3 where we have the index)
    // A query comes in and the index is not in the cache -> we need to load the index from s3 based on the segment files id

    pub(crate) fn create(
        &self,
        // TODO: This should not take Segment. The index layer should not know about the segment concept
        segment: &Segment,
        dimensionality: i32,
    ) -> Result<Arc<RwLock<HnswIndex>>, Box<dyn ChromaError>> {
        let id = Uuid::new_v4();
        let index_storage_path = self.temporary_storage_path.join(id.to_string());
        self.create_dir_all(&index_storage_path)?;
        let index_config = IndexConfig::from_segment(&segment, dimensionality);
        let hnsw_config = HnswIndexConfig::from_segment(segment, &index_storage_path)?;
        let mut cache = self.cache.write();
        let index = Arc::new(RwLock::new(HnswIndex::init(
            &index_config,
            Some(&hnsw_config),
            id,
        )?));
        cache.insert(id, index.clone());
        Ok(index)
    }

    pub(crate) fn commit(&self, id: &Uuid) -> Result<(), Box<dyn ChromaError>> {
        let cache = self.cache.read();
        let index = match cache.get(id) {
            Some(index) => index,
            None => {
                // TODO: error
                panic!("Trying to commit index that doesn't exist");
            }
        };
        index.write().save()?;
        Ok(())
    }

    pub(crate) async fn flush(&self, id: &Uuid) -> Result<(), Box<dyn ChromaError>> {
        // Scope to drop the cache lock before we await to write to s3
        // TODO: since we commit(), we don't need to save the index here
        {
            let cache = self.cache.read();
            let index = match cache.get(id) {
                Some(index) => index,
                None => {
                    // TODO: error
                    panic!("Trying to flush index that doesn't exist");
                }
            };
            index.write().save()?;
        }
        let index_storage_path = self.temporary_storage_path.join(id.to_string());
        for file in FILES.iter() {
            let file_path = index_storage_path.join(file);
            let key = self.format_key(id, file);
            let res = self
                .storage
                .put_file(&key, file_path.to_str().unwrap())
                .await;
            match res {
                Ok(_) => {
                    println!("Flushed hnsw index file: {}", file);
                }
                Err(e) => {
                    // TODO: return err
                    panic!("Failed to flush index: {}", e);
                }
            }
        }
        Ok(())
    }

    fn create_dir_all(&self, path: &PathBuf) -> Result<(), Box<HnswIndexProviderFileError>> {
        match std::fs::create_dir_all(path) {
            Ok(_) => Ok(()),
            Err(e) => return Err(Box::new(HnswIndexProviderFileError::IOError(e))),
        }
    }
}

#[derive(Error, Debug)]
pub(crate) enum HnswIndexProviderOpenError {
    #[error("Index configuration error")]
    IndexConfigError(#[from] IndexConfigFromSegmentError),
    #[error("Hnsw index file error")]
    FileError(#[from] HnswIndexProviderFileError),
}

impl ChromaError for HnswIndexProviderOpenError {
    fn code(&self) -> ErrorCodes {
        match self {
            HnswIndexProviderOpenError::IndexConfigError(e) => e.code(),
            HnswIndexProviderOpenError::FileError(_) => ErrorCodes::Internal,
        }
    }
}

#[derive(Error, Debug)]
pub(crate) enum HnswIndexProviderFileError {
    #[error("IO Error")]
    IOError(#[from] std::io::Error),
}
