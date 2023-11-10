package tech.molecules.leet.chem.sar.analysis;

import org.apache.commons.lang3.tuple.Pair;

import java.util.*;
import java.util.stream.Collectors;

public class DefaultPart implements Part {

    private List<Pair<String, String>> data;
    private List<String> complement;

    private int hash = -1;

    public DefaultPart(List<Pair<String, String>> data, List<String> complement) {
        this.initialize(data,complement);
    }

    @Override
    public void initialize(List<Pair<String, String>> parts, List<String> complement) {
        this.data = new ArrayList<>(parts);
        this.complement = new ArrayList<>(complement);
        recomputeHash();
    }

    private void recomputeHash() {
        Map<String,String> hm     = new HashMap<>();
        data.stream().forEach( xi -> hm.put(xi.getLeft(),xi.getRight()));
        complement.stream().forEach( xi -> hm.put(xi,""));
        this.hash = hm.hashCode();
    }

    /**
     *
     * @return hashmap with all elements, also with complement labels (they have empty string)
     */
    public Map<String,String> getFullMap() {
        Map<String,String> hm     = new HashMap<>();
        data.stream().forEach( xi -> hm.put(xi.getLeft(),xi.getRight()));
        complement.stream().forEach( xi -> hm.put(xi,""));
        return hm;
    }

    @Override
    public List<String> getPartLabels() {
        return data.stream().map( xi -> xi.getLeft() ).collect(Collectors.toList());
    }

    @Override
    public List<String> getComplementLabels() {
        return new ArrayList<>(this.complement);
    }

    @Override
    public List<Pair<String, String>> getVariants() {
        return new ArrayList<>(this.data);
    }

    @Override
    public int hashCode() {
        return this.hash;
    }

    @Override
    public boolean equals(Object obj) {
        if(obj instanceof Part) {
            Part pobj = (Part) obj;
            return this.getFullMap().equals(pobj.getFullMap());
        }
        return false;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        Map<String,String> hm     = new HashMap<>();
        data.stream().forEach( xi -> sb.append(xi.getLeft()+"->"+xi.getRight()));
        complement.stream().forEach( xi -> sb.append(xi+"->Complement"));
        return sb.toString();
    }

}
