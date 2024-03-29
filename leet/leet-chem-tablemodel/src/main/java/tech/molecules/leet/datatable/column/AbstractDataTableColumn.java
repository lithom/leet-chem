package tech.molecules.leet.datatable.column;

import tech.molecules.leet.datatable.DataProvider;
import tech.molecules.leet.datatable.DataTableColumn;
import tech.molecules.leet.datatable.NumericDatasource;

import java.awt.*;
import java.io.Serializable;
import java.util.List;
import java.util.function.Function;

public abstract class AbstractDataTableColumn<T,V> implements DataTableColumn<T,V> , Serializable {

    // Explicit serialVersionUID for interoperability
    private static final long serialVersionUID = 1L;

    transient private DataProvider<T> dp = null;
    private Class<V> representationClass;

    public AbstractDataTableColumn(Class<V> representationClass) {
        this.representationClass = representationClass;
    }

    public AbstractDataTableColumn(Class<V> representationClass, DataProvider<T> dp) {
        this.representationClass = representationClass;
        this.setDataProvider(dp);
    }

    @Override
    public Class<V> getRepresentationClass() {
        return this.representationClass;
    }



    public AbstractDataTableColumn<T,V> getThisColumn() {return this;}

    public abstract V processData(T data);

    private static class DefaultDataProviderListener implements DataProvider.DataProviderListener {
        private DataTableColumn col;
        private DataTableColumnListenerHelper dpListener;
        public DefaultDataProviderListener(DataTableColumn col) {
            this.col = col;
            this.dpListener = new DataTableColumnListenerHelper(col);
        }
        @Override
        public void dataChanged(List<String> keysChanged) {
            //col.listenerHelper.fireDataProviderChanged(getThisColumn(),getThisColumn().dp);
            this.dpListener.fireDataProviderChanged(col.getDataProvider());
        }
    }

    transient private DefaultDataProviderListener defaultDataProviderListener;

//    private DataProvider.DataProviderListener dataProviderListener = new DataProvider.DataProviderListener() {
//        @Override
//        public void dataChanged(List<String> keysChanged) {
//            listenerHelper.fireDataProviderChanged(getThisColumn(),getThisColumn().dp);
//        }
//    };

    @Override
    public void setDataProvider(DataProvider<T> dp) {
        this.dp = dp;
        if(this.defaultDataProviderListener!=null) {
            dp.removeDataProviderListener(this.defaultDataProviderListener);
        }
        this.defaultDataProviderListener = new DefaultDataProviderListener(this);
        dp.addDataProviderListener(this.defaultDataProviderListener);
        //dp.removeDataProviderListener(this.dataProviderListener);
        //dp.removeDataProviderListener(this.dataProviderListener);
        //dp.addDataProviderListener(this.dataProviderListener);
    }

    public DataProvider<T> getDataProvider() {
        return this.dp;
    }

    @Override
    public CellValue<V> getValue(String key) {
        if(dp==null) {return new CellValue<V>(null,null);}
        V val = processData( dp.getData(key) );
        Color vcol = null;
        if( this.backgroundNumericDatasource!=null && this.backgroundColormap!=null) {
            vcol = this.backgroundColormap.apply(this.backgroundNumericDatasource.evaluate(val));
        }
        return new CellValue<V>(val,vcol);
    }

    @Override
    public V getRawValue(String key) {
        return processData( dp.getData(key) );
    }

    private Function<Double,Color> backgroundColormap = null;
    private NumericDatasource<V>   backgroundNumericDatasource = null;


    @Override
    public void setBackgroundColor(NumericDatasource nd, Function<Double, Color> colormap) {
        this.backgroundColormap = colormap;
        this.backgroundNumericDatasource = nd;
    }

    transient private DataTableColumnListenerHelper listenerHelper = new DataTableColumnListenerHelper(this);

    @Override
    public void addColumnListener(DataTableColumnListener li) {
        this.listenerHelper.addColumnListener(li);
    }

    @Override
    public boolean removeColumnListener(DataTableColumnListener li) {
        return this.listenerHelper.removeColumnListener(li);
    }
}
