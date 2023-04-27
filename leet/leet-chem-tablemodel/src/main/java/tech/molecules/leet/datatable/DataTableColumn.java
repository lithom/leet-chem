package tech.molecules.leet.datatable;

import org.apache.commons.lang3.tuple.Pair;

import javax.swing.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public interface DataTableColumn<T,U> {

    public void setDataProvider(DataProvider<T> dp);

    public U getValue(String key);

    default public List<NumericDatasource> getNumericDatasources() {
        return new ArrayList<>();
    }

    default public List<DataRepresentation> getAlternativeRepresentations() {
        return new ArrayList<>();
    }

    default public List<Pair<List<String>,AbstractColumnSpecificAction<U>>> getColumnSpecificActions() {
        return new ArrayList<>();
    }

    default public List<Pair<List<String>,AbstractColumnRowSpecificAction<U>>> getColumnRowSpecificActions() {
        return new ArrayList<>();
    }

    default public List<DataFilter<U>> getFilters() {
        return new ArrayList<>();
    }

    default public List<DataSort<U>> getSortMethods() {
        return new ArrayList<>();
    }

    default public Map<DataRepresentation,List<Pair<List<String>,AbstractColumnSpecificAction<U>>>> createDataRepresentationBasedColumnSpecificActions() {
        return new HashMap<>(); // TODO implement..
    }


    public interface DataTableColumnListener {
        public void filteringChanged();
    }

    public void addColumnListener(DataTableColumnListener li);
    public boolean removeColumnListener(DataTableColumnListener li);

}
