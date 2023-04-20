package tech.molecules.leet.datatable.chem;

import tech.molecules.leet.chem.StructureWithID;
import tech.molecules.leet.datatable.DataProvider;
import tech.molecules.leet.datatable.DataTableColumn;
import tech.molecules.leet.datatable.NumericDatasource;

import java.util.ArrayList;
import java.util.List;

public class StructureWithIDColumn implements DataTableColumn<StructureWithID, StructureWithID> {

    private DataProvider<StructureWithID> dp;

    @Override
    public void setDataProvider(DataProvider<StructureWithID> dp) {
        this.dp = dp;
    }

    @Override
    public StructureWithID getValue(String key) {
        return this.dp.getData(key);
    }

    @Override
    public List<NumericDatasource> getNumericDatasources() {
        return new ArrayList<>();
    }

    @Override
    public void addColumnListener(DataTableColumnListener li) {

    }

    @Override
    public boolean removeColumnListener(DataTableColumnListener li) {
        return false;
    }
}
