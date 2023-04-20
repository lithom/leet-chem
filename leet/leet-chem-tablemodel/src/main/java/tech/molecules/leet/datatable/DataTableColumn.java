package tech.molecules.leet.datatable;

import java.util.List;

public interface DataTableColumn<T,U> {

    public void setDataProvider(DataProvider<T> dp);

    public U getValue(String key);

    public List<NumericDatasource> getNumericDatasources();

    public interface DataTableColumnListener {
        public void filteringChanged();
    }

    public void addColumnListener(DataTableColumnListener li);
    public boolean removeColumnListener(DataTableColumnListener li);

}
