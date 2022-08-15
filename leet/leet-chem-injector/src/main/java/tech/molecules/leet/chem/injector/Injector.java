package tech.molecules.leet.chem.injector;

import com.fasterxml.jackson.core.JsonProcessingException;
import tech.molecules.leet.chem.LeetSerialization;
import tech.molecules.leet.chem.mutator.FragmentDecompositionSynthon;
import tech.molecules.leet.chem.mutator.SimpleSynthonWithContext;
import tech.molecules.leet.chem.mutator.SynthonWithContext;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Injector {

    // map from bidir
    private Map<String, List<String>> fragments = new HashMap<>();

    public static Injector INJECTOR = null;

    public static void initInjector(List<String> synthonsWithContext) {
        INJECTOR = new Injector(synthonsWithContext);
    }

    public Injector(List<String> synthonsWithContext) {
        // init:
        this.fragments = new HashMap<>();

        for(String si : synthonsWithContext) {
            SimpleSynthonWithContext fdsi = null;
            try {
                fdsi = LeetSerialization.OBJECT_MAPPER.readValue(si, SimpleSynthonWithContext.class);
            } catch (JsonProcessingException e) {
                throw new RuntimeException(e);
            }

            String context1 = fdsi.getContextBidirectirectional(1,1).getIDCode();
            if(!fragments.containsKey(context1)) {
                fragments.put(context1,new ArrayList<>());
            }
            fragments.get(context1).add(si);
        }
    }


    public List<String> getSimpleSynthonsForContext(String context1) {
        return this.fragments.get(context1);
    }

}
