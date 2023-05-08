package tech.molecules.leet.datatable.column;

import tech.molecules.leet.datatable.DataProvider;
import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.DataTableColumn;

import java.util.ArrayList;
import java.util.List;

public class DataTableColumnListenerHelper {

    private List<DataTableColumn.DataTableColumnListener> listeners = new ArrayList<>();

    public void fireFilteringChanged(DataTableColumn col) {
        for(DataTableColumn.DataTableColumnListener li : this.listeners) {
            li.filteringChanged(col);
        }
    }

    public void fireSortingChanged(DataTableColumn col) {
        for(DataTableColumn.DataTableColumnListener li : this.listeners) {
            li.sortingChanged(col);
        }
    }

    public void fireDataProviderChanged(DataTableColumn col, DataProvider ndp) {
        for(DataTableColumn.DataTableColumnListener li : this.listeners) {
            li.dataProviderChanged(col,ndp);
        }
    }


    public void addColumnListener(DataTableColumn.DataTableColumnListener li) {
        this.listeners.add(li);
    }

    public boolean removeColumnListener(DataTableColumn.DataTableColumnListener li) {
        return this.listeners.remove(li);
    }

}
