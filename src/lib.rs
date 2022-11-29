use std::collections::{
    HashMap,
    hash_map::Iter,
};
use std::iter::Peekable;
use std::str::Split;

const DELIMITER: char = '/';
const SINGLE_LEVEL_WILDCARD: char = '+';
const MULTI_LEVEL_WILDCARD: char = '#';
const DOLLAR: char = '$';

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Word {
    /// Single level wildcard
    Plus,
    /// Multi level wildcard
    Num,
    /// Regular level
    Regular(String),
}

// #[derive(Debug, Default, Clone, PartialEq, Eq)]
type Children = HashMap<(Word, NodeType), Box<Node>>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum NodeType {
    Leaf,
    Branch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Leaf{
        word: Word,
    },
    Branch{
        word: Word,
        children: Children,
    },
}

/// A trie for MQTT topic filters
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Trie {
    len: usize,
    root: Children,
}

impl<'a> Iterator for ChildrenIter<'a> {
    type Item = (&'a Word, &'a Node);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|((k, _), v)| (k, v.as_ref()))
    }
}

impl<'a> Iterator for NodeIter<'a> {
    type Item = (&'a Word, &'a Node);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            NodeIter::Leaf(mut node) => {
                node.take().map(|node| (node.word(), node))
            },
            NodeIter::Branch(iter) => iter.next(),
        }
    }
}

impl<T> From<T> for Word
where
    T: AsRef<str>,
{
    fn from(src: T) -> Self {
        let s = src.as_ref();
        match s.chars().next() {
            Some(SINGLE_LEVEL_WILDCARD) => Word::Plus,
            Some(MULTI_LEVEL_WILDCARD) => Word::Num,
            _ => Word::Regular(s.to_string()),
        }
    }
}

impl std::borrow::Borrow<str> for &Word {
    fn borrow(&self) -> &str {
        match self {
            Word::Plus => "+",
            Word::Num => "#",
            Word::Regular(s) => s,
        }
    }
}

impl Node {
    fn iter(&self) -> NodeIter<'_> {
        match self {
            Node::Leaf{..} => NodeIter::Leaf(Some(self)),
            Node::Branch{children, ..} => NodeIter::Branch(ChildrenIter(children.iter())),
        }
    }

    fn word(&self) -> &Word {
        match self {
            Node::Leaf{word, ..} => word,
            Node::Branch{word, ..} => word,
        }
    }

    fn is_wildcard(&self) -> bool {
        match self {
            Node::Leaf{word: Word::Plus | Word::Num} |
            Node::Branch{word: Word::Plus | Word::Num, ..} => true,
            _ => false,
        }
    }
}

impl<S> FromIterator<S> for Trie
where
    S: AsRef<str>,
{
    fn from_iter<T: IntoIterator<Item=S>>(iter: T) -> Self {
        let mut trie = Trie::default();

        for topic in iter.into_iter() {
            trie.insert(topic.as_ref());
        }

        trie
    }
}

impl Trie {

    /// Insert a MQTT topic into the trie
    pub fn insert(&mut self, topic: &str) {
        let mut words = topic
            .split(DELIMITER)
            .map(|s| Word::from(s))
            .peekable();

        let mut current = &mut self.root;

        while let Some(word) = words.next() {
            let is_branch = words.peek().is_some();
            let node_type = if is_branch { NodeType::Branch } else { NodeType::Leaf };
            let key = (word.clone(), node_type);

            let child = current.entry(key).or_insert_with(|| {
                if is_branch {
                    Box::new(Node::Branch {
                        word,
                        children: Default::default(),
                    })
                } else {
                    Box::new(Node::Leaf {
                        word,
                    })
                }
            });

            match **child {
                Node::Leaf{..} => {
                    break;
                },
                Node::Branch{ref mut children, ..} => {
                    current = children;
                },
            }
        }

        self.len += 1;
    }

    /// Returns the number of keys in the trie.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn match_topic<'a, 'b>(&'a self, topic: &'b str) -> TrieMatcher<'b>
    where
        'a: 'b,
    {
        let mut levels = topic.split(DELIMITER).peekable();
        let current_level = levels.next();
        let inner = InnerMatcher {
            first_level: true,
            levels,
            current_level,
            children: NodeIter::Branch(ChildrenIter(self.root.iter())),
        };

        TrieMatcher {
            starts_with_dollar: topic.starts_with(DOLLAR),
            path: Default::default(),
            stack: vec![inner],
        }
    }
}

struct InnerMatcher<'a> {
    first_level: bool,
    // Topic level iterator
    levels: Peekable<Split<'a, char>>,
    current_level: Option<&'a str>,
    children: NodeIter<'a>,
}

struct ChildrenIter<'a>(Iter<'a, (Word, NodeType), Box<Node>>);

enum NodeIter<'a> {
    Leaf(Option<&'a Node>),
    Branch(ChildrenIter<'a>),
}

pub struct TrieMatcher<'a> {
    starts_with_dollar: bool,
    path: Vec<&'a Word>,
    stack: Vec<InnerMatcher<'a>>,
}

impl<'a> Iterator for TrieMatcher<'a> {
    type Item = Vec<&'a Word>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // NOTE: If the children iterators are empty, we have reached the end of the trie
            let current = self.stack.last_mut()?;

            let (_word, node) = if let Some(next) = current.children.next() {
                next
            } else {
                // Has no more siblings
                let _ = self.stack.pop();
                let _ = self.path.pop();
                continue;
            };

            let end_of_topic = current.levels.peek().is_none();
            let curr_level = current.current_level;

            use Node::*;
            match (node, curr_level, end_of_topic) {

                // Not match
                //
                // NOTE:According to the MQTT spec, the topic starts with a $ is a system topic,
                // and MUST NOT be matched Topic Filters starting with a wildcard character (# or +).
                //
                // See http://docs.oasis-open.org/mqtt/mqtt/v3.1.1/os/mqtt-v3.1.1-os.html#_Toc398718108
                (node, _, _) if current.first_level && node.is_wildcard() && self.starts_with_dollar => { }

                // Plus wildcard match any word in this level
                // NOTE: Matched
                (Leaf{word: word @ Word::Plus}, Some(_), true) => {
                    let mut result = self.path.clone();
                    result.push(word);
                    return Some(result);
                }

                // Number sign wildcard match any word in this level and all the following levels
                // NOTE: Matched all the following levels
                // WARN: Number sign wildcard must only be on the leaf node
                (Leaf{word: word @ Word::Num}, _, _) => {
                    let mut result = self.path.clone();
                    result.push(word);
                    return Some(result);
                }

                // NOTE: Matched regular word on the leaf node
                (Leaf{word: word @ Word::Regular(s)}, Some(curr_level), true) if s == curr_level => {
                    let mut result = self.path.clone();
                    result.push(word);
                    return Some(result);
                }

                // Match branch, go deep
                (Branch{word: word @ Word::Plus, .. }, Some(_), false) => {
                    let mut next_levels = current.levels.clone();
                    let next = next_levels.next();
                    self.path.push(word);
                    self.stack.push(InnerMatcher {
                        first_level: false,
                        levels: next_levels,
                        current_level: next,
                        children: node.iter(),
                    });
                }

                // NOTE: Matched the current level but need to CONTINUE to try the next level
                (Branch{word: word @ Word::Regular(s), ..}, Some(curr_level), _) if s == curr_level => {
                    // TODO: save the current cursor
                    let mut next_levels = current.levels.clone();
                    let next = next_levels.next();
                    self.path.push(word);
                    self.stack.push(InnerMatcher {
                        first_level: false,
                        levels: next_levels,
                        current_level: next,
                        children: node.iter(),
                    });
                }

                (Branch{word: Word::Num, ..}, _, _) => {
                    unreachable!("Number sign wildcard must only be on the leaf node");
                }

                // NOTE: Not matched
                _ => { }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_prefix() {
        let mut trie = Trie::default();
        trie.insert("aa/bb/cc");
        trie.insert("aa/+/cc");
        trie.insert("#");

        assert_eq!(trie.len(), 3);

        let mut result = trie.match_topic("aa/bb/cc").map(|m| m.join("/")).collect::<Vec<_>>();
        result.sort();
        assert_eq!(result, vec!["#", "aa/+/cc", "aa/bb/cc"]);
    }

    #[test]
    fn from_iter() {
        let topic_filters = [
            "aa/bb/cc",
            "aa/+/cc",
            "#",
            "aa/bb/cc/dd",
            "aa/bb/cc/+/dd",
        ];

        let trie = Trie::from_iter(topic_filters.iter());
        assert_eq!(trie.len(), 5);
    }

    #[test]
    fn empty() {
        let trie = Trie::default();
        assert_eq!(trie.len(), 0);
    }

    #[test]
    fn not_match() {
        let trie = Trie::from_iter(["aa/bb/cc", "cc/dd/ee", "+/bb"].iter());
        assert_eq!(trie.len(), 3);

        let matched = trie.match_topic("bb/cc/dd").collect::<Vec<_>>();
        assert!(matched.is_empty());
    }

    #[test]
    fn plus() {
        let trie = Trie::from_iter(["aa/bb/cc", "cc/dd/ee", "aa/+/bb", "aa/+"].iter());

        let mut matched = trie.match_topic("aa/bb").map(|m| m.join("/")).collect::<Vec<_>>();
        matched.sort();

        assert_eq!(matched, vec!["aa/+"]);

        let trie = Trie::from_iter(["aa/bb/cc", "cc/dd/ee", "aa/+/bb", "aa/+"].iter());

        let mut matched = trie.match_topic("aa/bb/bb").map(|m| m.join("/")).collect::<Vec<_>>();
        matched.sort();

        assert_eq!(matched, vec!["aa/+/bb"]);
    }
}
