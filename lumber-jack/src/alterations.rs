/*
    File meant to contain common data pre-processing utilities
*/

use std::vec::Vec;
use std::collections::{HashMap, BTreeSet};


pub fn split_n_hot_encode<'a>(array_of_strings: &mut Vec<&'a str>, sep: &str, cutoff: usize) -> (Vec<Vec<u8>>, Vec<&'a str>) {
    /*
    Given an array of strings of size (n_samples,) will return a one-hot encoded matrix for each sample
    indicating if the string/word was present
    */

    // The mapping of unique strings to how many times they occur
    let mut string_counts: HashMap<&str, usize> = HashMap::new();

    // HashMap of strings and their counts
    for string in array_of_strings.iter() {
        let count = string_counts.entry(string).or_insert(0);
        *count += 1;
    }

    // Remove keys whose value is below cutoff vvalue, returns immediately if cutoff < 1
    let string_counts: HashMap<&str, usize> = prune_keys(string_counts, cutoff);
    let key_words: Vec<&str> = string_counts.keys().cloned().collect();

    // Create one-hot matrix
    let matrix: Vec<Vec<u8>> = produce_onehot(&key_words, &array_of_strings);


    let array_of_strings: Vec<&str> = string_counts.keys().cloned().collect();
    (matrix, array_of_strings)

}

fn produce_onehot(key_words: &Vec<&str>, raw_texts: &Vec<&str>) -> Vec<Vec<u8>> {
    /*
    Given an array of raw texts and an array of words to look for, return one-hot matrix
    indicating if word at each index of key_words occurs in raw_text array

    Parameters
    ----------
    key_words: Array of keywords to concern oneself about in searching for
    raw_text:  Array of raw text strings to search keywords for

    Returns
    -------
    2d array where each occurrence of raw_text has a vector matching key_words length and order
    and consists of binary indicators if the key_word was present in the instance of raw_text
    */

    let mut matrix: Vec<Vec<u8>> = Vec::with_capacity(raw_texts.len());

    // This portion could be done parallel by doing each raw text by itself and then collecting
    // all resulting one-hot vectors
    for (i, raw_text) in raw_texts.iter().enumerate() {
        for (j, key_word) in key_words.iter().enumerate() {
            if raw_texts.contains(key_word) {
                matrix[i][j] = 1;
            } else {
                matrix[i][j] = 0;
            }
        }
    }
    matrix
}

fn prune_keys(mut string_counts: HashMap<&str, usize>, cutoff: usize) -> HashMap<&str, usize> {
    /*

    Handles the removal of keys from a Hashmap given a cutoff value
    If the key's value is less than that value, the key is removed from the HashMap

    Parameters
    ----------
    string_counts:  HashMap consisting of string keys and a count of that string's occurrences.
    cutoff:         Value which the key's value must be over in order to keep.

    Returns
    -------
    HashMap which has keys removed whose values were below the cutoff
    */

    // If cutoff is 0, then there is no point other than to return original mapping
    if cutoff < 1 {
        return string_counts
    }

    // Define keys to remove set in this scope
    let mut keys_to_remove: BTreeSet<&str> = BTreeSet::new();

    // Iterate over strings and their occurrence counts, adding to keys_to_remove as needed
    for string in string_counts.keys() {
        let count = string_counts.get(string);
        if let Some(ct) = count {
            if ct < &cutoff {
                keys_to_remove.insert(string);
            }
        }
    }

    // Remove all keys placed into keys_to_remove
    for key in keys_to_remove.iter() {
        string_counts.remove(key);
    }

    // Return the pruned string counts HashMap
    string_counts
}