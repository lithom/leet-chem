package tech.molecules.leet.table.chem;

import tech.molecules.leet.chem.mutator.SynthonWithContext;
import tech.molecules.leet.table.NDataProvider;
import tech.molecules.leet.table.NStructureDataProvider;

public interface NSynthonWithContextDataProvider extends NDataProvider {

    public static class SynthonStructureWithID extends NStructureDataProvider.StructureWithID {
        private SynthonWithContext synthon;
        public SynthonStructureWithID(SynthonWithContext synth) {
            super(synth.getSynthon().getIDCode(),"",new String[]{ synth.getSynthon().getIDCode() , synth.getSynthon().getIDCoordinates() });
            this.synthon = synthon;
        }
        public SynthonWithContext getSynthon() {
            return this.synthon;
        }
    }

    public SynthonWithContext getStructureData(String rowid);
}
