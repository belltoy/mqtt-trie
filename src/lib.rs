#![doc = include_str!("../README.md")]

//! A Trie implementation for MQTT Topic Filters.

mod trie;

#[doc(inline)]
pub use trie::Trie;
#[doc(inline)]
pub use trie::TrieMatcher;
