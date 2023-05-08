package tech.molecules.leet.datatable.chem;

import tech.molecules.leet.chem.StructureWithID;
import tech.molecules.leet.datatable.DataProvider;
import tech.molecules.leet.datatable.DataTableColumn;
import tech.molecules.leet.datatable.NumericDatasource;
import tech.molecules.leet.datatable.column.AbstractDataTableColumn;

import java.util.ArrayList;
import java.util.List;

public class StructureWithIDColumn extends AbstractDataTableColumn<StructureWithID,StructureWithID> {

    @Override
    public StructureWithID processData(StructureWithID data) {
        return data;
    }
}
