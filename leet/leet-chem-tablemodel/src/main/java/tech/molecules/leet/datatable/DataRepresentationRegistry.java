package tech.molecules.leet.datatable;

import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class DataRepresentationRegistry {

    private static DataRepresentationRegistry INSTANCE;

    public static DataRepresentationRegistry getInstance() {
        if(INSTANCE == null) {
            INSTANCE = new DataRepresentationRegistry();
        }
        return INSTANCE;
    }

    /**
     * Map from original class to alternative representations
     * access synchronized via "this"
     */
    private Map<Class, List<DataRepresentation>> representations;

    public synchronized <T> void registerRepresentation(Class<T> original, DataRepresentation<T,?> representation) {
        if(!representations.containsKey(original)) {representations.put(original,new ArrayList<>());}
        representations.get(original).add(representation);
    }

    public List<Pair<Class,DataRepresentation>> findRepresentations(Class original) {
        List<Pair<Class,DataRepresentation>> alternatives = new ArrayList<>();
        for( Class ci : original.getClasses()) {
            List<DataRepresentation> di = null;
            synchronized(this) {
                di = representations.get(ci);
            }
            if(di!=null) {
                for(DataRepresentation dri : di) {
                    alternatives.add(Pair.of(ci,dri));
                }
            }
        }
        return alternatives;
    }

}
