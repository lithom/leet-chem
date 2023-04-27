package tech.molecules.leet.datatable;

import org.apache.commons.lang3.tuple.Pair;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class DataRepresentationHelper {


    public static List<Pair<Class,DataRepresentation>> findSpecificRepresentations(DataRepresentationRegistry registry, Class original, Class requiredInterface) {
        return registry.findRepresentations(original).stream().filter( pi -> Arrays.asList( pi.getRight().getRepresentationClass().getClasses() ).contains(requiredInterface) ).collect(Collectors.toList());
    }

    public static List<Pair<Class,DataRepresentation>> findSpecificRepresentations(DataRepresentationRegistry registry, Class original, List<Class> requiredInterfaces) {
        return registry.findRepresentations(original).stream().filter( pi -> Arrays.asList( pi.getRight().getRepresentationClass().getClasses() ).containsAll(requiredInterfaces) ).collect(Collectors.toList());
    }

    public static List<Pair<Class,DataRepresentation>> findNumericRepresentations(DataRepresentationRegistry registry, Class original) {
        //Class[] numeric = new Class[]{Integer.class,Double.class,Float.class};
        return findSpecificRepresentations(registry,original,Double.class);
    }


}
