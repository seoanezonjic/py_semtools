class FileParser:
    #############################################
    # FIELDS
    #############################################
    # Handled class variables
    # => basic_tags :: hash with main OBO structure tags
    # => tags_with_trailing_modifiers :: tags which can include extra info after specific text modifiers
    basic_tags = {'ancestors': ['is_a'], 'obsolete': 'is_obsolete', 'alternative': ['replaced_by', 'consider', 'alt_id']}
    tags_with_trailing_modifiers = ['is_a', 'union_of', 'disjoint_from', 'relationship', 'subsetdef', 'synonymtypedef', 'property_value']
    multivalue_tags = ['alt_id', 'is_a', 'subset', 'synonym', 'xref', 'intersection_of', 'union_of', 'disjoint_from', 'relationship', 'replaced_by', 'consider', 'subsetdef', 'synonymtypedef', 'property_value', 'remark', 'namespace'] # namespace only is used by EFO

