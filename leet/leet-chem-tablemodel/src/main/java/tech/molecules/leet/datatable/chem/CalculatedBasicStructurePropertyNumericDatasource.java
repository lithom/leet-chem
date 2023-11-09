package tech.molecules.leet.datatable.chem;

import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.StructureRecord;
import tech.molecules.leet.chem.StructureWithID;
import tech.molecules.leet.chem.mutator.properties.ChemPropertyCounts;
import tech.molecules.leet.datatable.AbstractNumericDatasource;
import tech.molecules.leet.datatable.DataTableColumn;
import tech.molecules.leet.datatable.NumericDatasource;

public class CalculatedBasicStructurePropertyNumericDatasource<T extends StructureWithID> extends AbstractNumericDatasource<T> {

    private int property;

    public CalculatedBasicStructurePropertyNumericDatasource(DataTableColumn<?, T> col, int property) {
        super(ChemPropertyCounts.COUNTS_ALL[property].name, col);
        this.property = property;
    }

    @Override
    public Double evaluate(T original) {
        return (double) ChemPropertyCounts.COUNTS_ALL[this.property].evaluator.apply(ChemUtils.parseIDCode(original.structure[0],original.structure[1]));
    }
}
