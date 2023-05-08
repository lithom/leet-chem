package tech.molecules.leet.chem.injector;

import tech.molecules.leet.chem.mutator.SynthonWithContext;
import tech.molecules.leet.table.NDataProvider;
import tech.molecules.leet.table.chem.NSynthonWithContextDataProvider;

import java.util.Map;

public class InjectorDatasetProvider implements NSynthonWithContextDataProvider {

    private Map<String,SynthonWithContext> synthons;

    public InjectorDatasetProvider(Map<String, SynthonWithContext> synthons) {
        this.synthons = synthons;
    }

    @Override
    public SynthonWithContext getStructureData(String rowid) {
        return null;
    }

}
