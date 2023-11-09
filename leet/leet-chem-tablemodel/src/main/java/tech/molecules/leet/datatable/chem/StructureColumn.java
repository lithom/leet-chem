package tech.molecules.leet.datatable.chem;

import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.StructureWithID;
import tech.molecules.leet.datatable.column.AbstractDataTableColumn;

public class StructureColumn extends AbstractDataTableColumn<StereoMolecule,StereoMolecule> {

    public StructureColumn() {
        super(StereoMolecule.class);
    }

    @Override
    public StereoMolecule processData(StereoMolecule data) {
        return data;
    }
}
