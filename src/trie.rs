use std::collections::HashMap;
use std::iter::Peekable;
use std::str::Split;

const DELIMITER: char = '/';
const DOLLAR: char = '$';
const SINGLE_LEVEL_WILDCARD: &str = "+";
const MULTI_LEVEL_WILDCARD: &str = "#";

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Level<T> {
    /// Single level wildcard
    Plus,
    /// Multi level wildcard
    Num,
    /// Regular level
    Regular(T),
}

type OwnedLevel = Level<String>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OwnedKey {
    value: OwnedLevel,
    node_type: NodeType,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum NodeType {
    Leaf,
    Branch,
}
impl Copy for NodeType {}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Leaf{
        value: OwnedLevel,
        /// The cached path associated with the leaf node.
        path: String,
    },
    Branch{
        value: OwnedLevel,
        children: Children,
    },
}

/// A trie to find MQTT topic filters stored in it by a given topic name.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Trie {
    len: usize,
    root: Children,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Children(HashMap<OwnedKey, Box<Node>>);

trait Key {
    fn key(&self) -> (Level<&str>, NodeType);
}

impl Key for OwnedKey {
    fn key(&self) -> (Level<&str>, NodeType) {
        (self.value.as_ref(), self.node_type)
    }
}

impl Key for (Level<&str>, NodeType) {
    fn key(&self) -> (Level<&str>, NodeType) {
        (self.0.clone(), self.1)
    }
}

impl Key for (&str, NodeType) {
    fn key(&self) -> (Level<&str>, NodeType) {
        (Level::Regular(self.0), self.1)
    }
}

impl<'a> std::borrow::Borrow<dyn Key + 'a> for OwnedKey {
    fn borrow(&self) -> &(dyn Key + 'a) {
        self
    }
}

impl<'a> Eq for dyn Key + 'a {}
impl<'a> PartialEq for dyn Key + 'a {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl<'a> std::hash::Hash for dyn Key + 'a {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

impl Level<String> {
    fn as_ref(&self) -> Level<&str> {
        use Level::*;
        match self {
            Plus => Plus,
            Num => Num,
            Regular(s) => Regular(s.as_str()),
        }
    }
}

impl<T> From<T> for OwnedLevel
where
    T: AsRef<str>,
{
    fn from(src: T) -> Self {
        match src.as_ref() {
            SINGLE_LEVEL_WILDCARD => Level::Plus,
            MULTI_LEVEL_WILDCARD => Level::Num,
            _ => Level::Regular(src.as_ref().to_string()),
        }
    }
}

struct ChildrenNodeMatcher<'a> {
    level: &'a str,
    end_of_topic: bool,
    children: &'a Children,
    search_type: NodeSearchType,
}

// The fixed search sequences for children nodes search.
#[derive(Debug)]
enum NodeSearchType {
    // search node of number sign #
    NumNode,
    // search leaf node of plus sign +
    PlusLeaf,
    // search branch node of plus sign +
    PlusBranch,
    // search leaf node of regular value
    RegularLeaf,
    // search branch node of regular value
    RegularBranch,
    // End of search
    End,
}

impl Children {
    /// Search children nodes by topic level value at the end of topic or at the middle level,
    /// returns an iterator which may result in a number sign node or plus sign node.
    ///
    /// We use HashMap as the children node container, so we only check 3 of them to
    /// determine the match result.
    ///
    /// TODO: Validate the query parameter.
    fn matches_nodes<'a, 'b>(&'a self, level: &'b str, end_of_topic: bool) -> ChildrenNodeMatcher<'b>
    where
        'a: 'b,
    {
        ChildrenNodeMatcher {
            level,
            end_of_topic,
            children: self,
            search_type: NodeSearchType::NumNode,
        }
    }
}

impl<'a> Iterator for ChildrenNodeMatcher<'a> {
    type Item = &'a Node;

    fn next(&mut self) -> Option<Self::Item> {
        use NodeSearchType::*;
        use Level::*;
        use NodeType::*;
        let children = &self.children.0;
        loop {
            // Search children nodes in the fixed order.
            let (rst, next) = if self.end_of_topic {
                // For leaf nodes iteration
                match &self.search_type {
                    NumNode => {
                        (children.get(&(Num, Leaf) as &dyn Key), PlusLeaf)
                    }
                    PlusLeaf => {
                        (children.get(&(Plus, Leaf) as &dyn Key), RegularLeaf)
                    }
                    // NOTE: DON'T clone string, borrow it
                    RegularLeaf => {
                        let key = (self.level, Leaf);
                        (children.get(&key as &dyn Key), End)
                    }
                    _ => {
                        return None;
                    }
                }
            } else {
                // For branch nodes iteration
                match &self.search_type {
                    NumNode => {
                        // NOTE: The number sign node is always at the leaf node
                        (children.get(&(Num, Leaf) as &dyn Key), PlusBranch)
                    }
                    PlusBranch => {
                        (children.get(&(Plus, Branch) as &dyn Key), RegularBranch)
                    }
                    // NOTE: DON'T clone string, borrow it
                    RegularBranch => {
                        let key = (self.level, Branch);
                        (children.get(&key as &dyn Key), End)
                    }
                    _ => {
                        return None;
                    }
                }
            };

            self.search_type = next;

            if let Some(rst) = rst {
                return Some(rst);
            }
        }
    }
}

impl Node {

    fn is_wildcard(&self) -> bool {
        match self {
            Node::Leaf{value: Level::Plus | Level::Num, ..} |
            Node::Branch{value: Level::Plus | Level::Num, ..} => true,
            _ => false,
        }
    }
}

impl Trie {
    /// Construct a new empty trie.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a MQTT `topic_filter` into the trie.
    pub fn insert(&mut self, topic_filter: &str) {
        let mut values = topic_filter
            .split(DELIMITER)
            .map(|s| Level::from(s))
            .peekable();

        let mut current = &mut self.root;

        while let Some(value) = values.next() {
            let is_branch = values.peek().is_some();
            let node_type = if is_branch { NodeType::Branch } else { NodeType::Leaf };
            let key = OwnedKey {
                value: value.clone(),
                node_type,
            };

            let child = current.0.entry(key).or_insert_with(|| {
                if is_branch {
                    Box::new(Node::Branch {
                        value,
                        children: Default::default(),
                    })
                } else {
                    Box::new(Node::Leaf {
                        value,
                        path: topic_filter.to_string(),
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

    /// Find all matched MQTT Topic Filters in the trie to the given `topic`.
    pub fn matches_topic<'a, 'b>(&'a self, topic: &'b str) -> TrieMatcher<'b>
    where
        'a: 'b,
    {
        let mut levels = topic.split(DELIMITER).peekable();
        // split an empty string has at least one element
        let current_level = levels.next();
        let end_of_topic = levels.peek().is_none();
        let inner = InnerMatcher {
            first_level: true,
            levels,
            current_level: current_level.clone(),
            // Since the str split iterator always has at least one element, it's safe to unwrap.
            children: self.root.matches_nodes(current_level.unwrap(), end_of_topic),
        };

        TrieMatcher {
            starts_with_dollar: topic.starts_with(DOLLAR),
            path: Default::default(),
            stack: vec![inner],
        }
    }
}

impl<S> FromIterator<S> for Trie
where
    S: AsRef<str>,
{
    /// Construct a `Trie` from an iterator over `&str`.
    ///
    /// TODO: It's better to validate the topic filter before insert it into the trie.
    fn from_iter<T: IntoIterator<Item=S>>(topic_filters: T) -> Self {
        let mut trie = Trie::default();

        for topic_filter in topic_filters.into_iter() {
            trie.insert(topic_filter.as_ref());
        }

        trie
    }
}

struct InnerMatcher<'a> {
    first_level: bool,
    // Topic level iterator
    levels: Peekable<Split<'a, char>>,
    current_level: Option<&'a str>,
    children: ChildrenNodeMatcher<'a>,
}

/// An iterator over the matched topic filters in the [`Trie`] via [`Trie::matches_topic`].
pub struct TrieMatcher<'a> {
    starts_with_dollar: bool,
    path: Vec<&'a OwnedLevel>,
    stack: Vec<InnerMatcher<'a>>,
}

impl<'a> Iterator for TrieMatcher<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // NOTE: If the children iterators are empty, we have reached the end of the trie
            let current = self.stack.last_mut()?;

            let node = if let Some(next) = current.children.next() {
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
                // NOTE: According to the MQTT spec, the topic starts with a $ is a system topic,
                // and MUST NOT be matched Topic Filters starting with a wildcard character (# or +).
                //
                // See http://docs.oasis-open.org/mqtt/mqtt/v3.1.1/os/mqtt-v3.1.1-os.html#_Toc398718108
                (node, _, _) if current.first_level && node.is_wildcard() && self.starts_with_dollar => { }

                // Plus wildcard match any value in this level
                // NOTE: Matched
                (Leaf{value: Level::Plus, path}, Some(_), true) => {
                    return Some(path.as_str());
                }

                // Number sign wildcard match any value in this level and all the following levels
                // NOTE: Matched all the following levels
                // WARN: Number sign wildcard must only be on the leaf node
                (Leaf{value: Level::Num, path}, _, _) => {
                    return Some(path.as_str());
                }

                // NOTE: Matched regular value on the leaf node
                (Leaf{value: Level::Regular(s), path}, Some(curr_level), true) if s == curr_level => {
                    return Some(path.as_str());
                }

                // Match branch, go deep
                (Branch{value: value @ Level::Plus, children}, Some(_), false) => {
                    let mut next_levels = current.levels.clone();
                    let next = next_levels.next();
                    let end_of_topic = next_levels.peek().is_none();
                    self.path.push(value);
                    self.stack.push(InnerMatcher {
                        first_level: false,
                        levels: next_levels,
                        current_level: next,
                        // Since current level is not at the end of the topic, it's safe to unwrap
                        children: children.matches_nodes(next.unwrap(), end_of_topic),
                    });
                }

                // NOTE: Matched the current level but need to CONTINUE to try the next level
                (Branch{value: value @ Level::Regular(s), children}, Some(curr_level), false) if s == curr_level => {
                    // NOTE: save the current cursor
                    let mut next_levels = current.levels.clone();
                    let next = next_levels.next();
                    let end_of_topic = next_levels.peek().is_none();
                    self.path.push(value);
                    self.stack.push(InnerMatcher {
                        first_level: false,
                        levels: next_levels,
                        current_level: next,
                        // Since current level is not at the end of the topic, it's safe to unwrap
                        children: children.matches_nodes(next.unwrap(), end_of_topic),
                    });
                }

                (Branch{value: Level::Num, ..}, _, _) => {
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

        let mut result = trie.matches_topic("aa/bb/cc").collect::<Vec<_>>();
        result.sort();
        assert_eq!(result, vec!["#", "aa/+/cc", "aa/bb/cc"]);
        assert!(trie.matches_topic("$SYS/foo").collect::<Vec<_>>().is_empty());

        let trie = Trie::from_iter(["aa/bb/cc", "#", "+/bb", "$SYS/#"].iter());
        assert_eq!(trie.len(), 4);
        assert_eq!(trie.matches_topic("$SYS/foo").collect::<Vec<_>>(), vec!["$SYS/#"]);

        let trie = Trie::from_iter(["sport/+"].iter());
        assert_eq!(trie.len(), 1);

        assert!(trie.matches_topic("sport").collect::<Vec<_>>().is_empty());
        assert_eq!(trie.matches_topic("sport/").collect::<Vec<_>>(), vec!["sport/+"]);
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

        let matched = trie.matches_topic("bb/cc/dd").collect::<Vec<_>>();
        assert!(matched.is_empty());
    }

    #[test]
    fn plus() {
        let trie = Trie::from_iter(["aa/bb/cc", "cc/dd/ee", "aa/+/bb", "aa/+"].iter());

        let mut matched = trie.matches_topic("aa/bb").collect::<Vec<_>>();
        matched.sort();

        assert_eq!(matched, vec!["aa/+"]);

        let trie = Trie::from_iter(["aa/bb/cc", "cc/dd/ee", "aa/+/bb", "aa/+"].iter());

        let mut matched = trie.matches_topic("aa/bb/bb").collect::<Vec<_>>();
        matched.sort();

        assert_eq!(matched, vec!["aa/+/bb"]);
    }
}
