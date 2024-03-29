package tech.molecules.leet.datatable;

import org.apache.commons.lang3.tuple.Pair;

import javax.swing.*;
import java.awt.*;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

public interface DataTableColumn<T,U> {

    public void setDataProvider(DataProvider<T> dp);

    public DataProvider<T> getDataProvider();

    public CellValue<U> getValue(String key);

    public U getRawValue(String key);

    default public List<NumericDatasource> getNumericDatasources() {
        return new ArrayList<>();
    }

    default public List<CategoricDatasource> getCategoricDatasources() { return new ArrayList<>(); }

    default public List<DataRepresentation> getAlternativeRepresentations() {
        return new ArrayList<>();
    }

    default public List<Pair<List<String>,AbstractColumnSpecificAction<U>>> getColumnSpecificActions() {
        return new ArrayList<>();
    }

    default public List<Pair<List<String>,AbstractColumnRowSpecificAction<U>>> getColumnRowSpecificActions() {
        return new ArrayList<>();
    }

    public Class<U> getRepresentationClass();

    default public List<DataFilter<U>> getFilters() {
        return new ArrayList<>();
    }

    default public List<DataSort<U>> getSortMethods() {
        return new ArrayList<>();
    }

    default public Map<DataRepresentation,List<Pair<List<String>,AbstractColumnSpecificAction<U>>>> createDataRepresentationBasedColumnSpecificActions() {
        return new HashMap<>(); // TODO implement..
    }

    public static class CellValue<U> implements Serializable {
        private static final long serialVersionUID = 1L;

        public final U val;
        public final Color colBG;
        public CellValue(U val, Color colBG) {
            this.val = val;
            this.colBG = colBG;
        }
    }

    public void setBackgroundColor(NumericDatasource nd, Function<Double, Color> colormap);

    public interface DataTableColumnListener {
        public void filteringChanged(DataTableColumn col);
        public void sortingChanged(DataTableColumn col);
        public void dataProviderChanged(DataTableColumn col, DataProvider newDP);
        public void visualizationChanged(DataTableColumn col);
    }

    public void addColumnListener(DataTableColumnListener li);
    public boolean removeColumnListener(DataTableColumnListener li);

}
