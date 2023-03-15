class JsonParser < FileParser

	def self.load(ontology, file, build: true)
        read(ontology, file) 
	end

   # Read a JSON file with an OBO_Handler object stored
    # ===== Parameters
    # +file+:: with object info
    # +file+:: if true, calculate indexes. Default: true
    # ===== Return
    # OBO_Handler internal fields 
    def self.read(ontology, file, build: true)
        # Read file
        jsonFile = File.open(file)
        jsonInfo = JSON.parse(jsonFile.read, :symbolize_names => true)
        # Pre-process (Symbolize some hashs values)
        jsonInfo[:terms].map{|id,info| symbolize_ids(info)} # STANZAS
        # Optional
        jsonInfo[:alternatives_index] = jsonInfo[:alternatives_index].map{|id,value| [id, value.to_sym]}.to_h unless jsonInfo[:alternatives_index].nil?
        jsonInfo[:ancestors_index].map {|id,family_arr| family_arr.map!{|item| item.to_sym}} unless jsonInfo[:ancestors_index].nil?
        jsonInfo[:descendants_index].map {|id,family_arr| family_arr.map!{|item| item.to_sym}} unless jsonInfo[:descendants_index].nil?
        jsonInfo[:dicts] = jsonInfo[:dicts].each do |flag, dictionaries|
            next if dictionaries.nil?
            # Special case: byTerm
            dictionaries[:byTerm] = dictionaries[:byTerm].map do |term, value| 
                if !term.to_s.scan(/\A[-+]?[0-9]*\.?[0-9]+\Z/).empty?  # Numeric dictionary
                    [term.to_s.to_i, value.map{|term| term.to_sym}]
                elsif value.is_a? Numeric # Numeric dictionary
                    [term.to_sym, value]
                elsif value.kind_of?(Array) && flag == :is_a
                    [term.to_sym, value.map{|v| v.to_sym}]
                else
                    [term.to_sym, value]
                end
            end
            dictionaries[:byTerm] = dictionaries[:byTerm].to_h
            # By value
            dictionaries[:byValue] = dictionaries[:byValue].map do |value, term| 
                if value.is_a? Numeric # Numeric dictionary
                    [value, term.to_sym]
                elsif term.is_a? Numeric # Numeric dictionary
                    [value.to_s.to_sym, term]
                elsif flag == :is_a
                    [value.to_sym, term.map{|v| v.to_sym}]
                elsif term.kind_of?(Array)
                    [value.to_sym, term.map{|t| t.to_sym}]
                else
                    [value.to_s, term.to_sym]
                end
            end
            dictionaries[:byValue] = dictionaries[:byValue].to_h
        end 
        if !jsonInfo[:profiles].nil?
            jsonInfo[:profiles].map{|id,terms| terms.map!{|term| term.to_sym}}
            jsonInfo[:profiles].keys.map{|id| jsonInfo[:profiles][id.to_s.to_i] = jsonInfo[:profiles].delete(id) if self.is_number?(id.to_s)}
        end
        jsonInfo[:removable_terms] = jsonInfo[:removable_terms].map{|term| term.to_sym} unless jsonInfo[:removable_terms].nil?
        jsonInfo[:items].each{|k,v| jsonInfo[:items][k] = v.map{|item| item.to_sym}} unless jsonInfo[:items].nil?
        jsonInfo[:term_paths].each{|term,info| jsonInfo[:term_paths][term][:paths] = info[:paths].map{|path| path.map{|t| t.to_sym}}} unless jsonInfo[:term_paths].nil?
        
        # Store info
        ontology.terms = jsonInfo[:terms]
        ontology.ancestors_index = jsonInfo[:ancestors_index]
        ontology.descendants_index = jsonInfo[:descendants_index]
        ontology.alternatives_index = jsonInfo[:alternatives_index]
        jsonInfo[:structureType] = jsonInfo[:structureType].to_sym unless jsonInfo[:structureType].nil?
        ontology.structureType = jsonInfo[:structureType]
        ontology.ics = jsonInfo[:ics]
        ontology.meta = jsonInfo[:meta]
        ontology.max_freqs = jsonInfo[:max_freqs]
        ontology.dicts = jsonInfo[:dicts]
        ontology.profiles = jsonInfo[:profiles]
        ontology.items = jsonInfo[:items]
        ontology.term_paths = jsonInfo[:term_paths]

        ontology.precompute() if build
    end

    def self.is_number? string
          true if Float(string) rescue false
    end

end
