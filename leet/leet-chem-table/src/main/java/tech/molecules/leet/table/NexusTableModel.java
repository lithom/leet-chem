package tech.molecules.leet.table;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;

import javax.swing.*;
import javax.swing.table.AbstractTableModel;
import java.util.*;
import java.util.stream.Collectors;

public class NexusTableModel extends AbstractTableModel {

    //private NexusOsirisDataset data;

    private List<String> allRows                 = new ArrayList<>();
    private List<String> visibleRows             = new ArrayList<>();
    //private List<Pair<NColumn, NStructureDataProvider>> columns = new ArrayList<>();

    /**
     * The data providers for the columns are specified in the columnDataProvider map.
     */
    private List<NColumn> columns = new ArrayList<>();

    private Map<NColumn,NDataProvider> columnDataProviders = new HashMap<>();

    Map<NColumn,List<NColumn.NexusRowFilter>> rowFilters = new HashMap<>();

//    public void setNexusData(NexusOsirisDataset data) {
//        this.data = data;
//        this.visibleRows = new ArrayList<>( data.getData().keySet() );
//    }

    public void setAllRows(List<String> rowids) {
        this.allRows = rowids;
        this.updateFiltering();
    }


    public List<String> getAllRows() {
        return this.allRows;
    }

    //public void setVisibleRows(List<String> rowids) {
    //    this.visibleRows = new ArrayList<>(rowids);
    //}

    public List<String> getVisibleRows() { return this.visibleRows; }

    private Map<NColumn, NColumn.ColumnDataListener> columnListeners = new HashMap<>();

    public <U,T> T getDatasetForColumn(NColumn<U,T> col) {
        //T dataset = null;
        //for(Pair<NColumn, NDataProvider> ci : this.columns) {
        //    if(ci.getLeft()==col) {return (U) ci.getRight();}
        //}
        return (T) this.columnDataProviders.get(col);
    }



    /**
     *
     * @param cols
     *
     */
    //public <U,T> void setNexusColumns(List<Pair<NColumn<U,T>, U>> cols) {
    public void setNexusColumnsWithDataProviders(List<Pair<NColumn, NDataProvider>> cols) {
        this.columns = new ArrayList<>( cols.stream().map( ci -> ci.getLeft() ).collect(Collectors.toList()) ) ;
        //List<NDataProvider> dps = new ArrayList<>( cols.stream().map( ci -> ci.getRight() ).collect(Collectors.toList()) ) ;;
        for(int zi=0;zi<this.columns.size();zi++) {
            NColumn cp = columns.get(zi);
            //NDataProvider dp = //this.columnDataProviders.get(cp);
            this.setDataProviderForColumn(cp,cols.get(zi).getRight());
            if(!columnListeners.containsKey(cp)) {
                NColumn.ColumnDataListener cdpl = new NColumn.ColumnDataListener() {
                    @Override
                    public void needFiltering() {
                        updateFiltering();
                    }
                };
                cp.addColumnDataListener(cdpl);
                columnListeners.put(cp,cdpl);
            }
        }
        this.updateFiltering();
        fireTableStructureChanged();
        fireNexusTableStructureChangedEvent();
    }

    public <U extends NDataProvider,T> void setDataProviderForColumn(NColumn<U,T> c, U np) {
        this.columnDataProviders.put(c,np);
    }

    public List<NColumn> getNexusColumns() {
        return this.columns;
    }

    @Override
    public int getRowCount() {
        return this.visibleRows.size();
    }

    @Override
    public int getColumnCount() {
        return this.columns.size();
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        //return columns.get(columnIndex).getData( columns.get(columnIndex).getRight(),visibleRows.get(rowIndex));
        return columns.get(columnIndex).getData( columnDataProviders.get(columns.get(columnIndex)),visibleRows.get(rowIndex));
    }

    public <U,T> void  addRowFilter(NColumn<U,T> col, NColumn.NexusRowFilter<U> filter ) {
        filter.setupFilter(this);
        if( !this.rowFilters.containsKey(col) ) {this.rowFilters.put(col,new ArrayList<>());}
        this.rowFilters.get(col).add(filter);
        this.updateFiltering();
    }

    @Override
    public boolean isCellEditable(int rowIndex, int columnIndex) {
        return true;
    }

    public void resetFilters() {
        this.rowFilters.clear();
    }

    public final Object lockTable = new Object();

    public void updateFiltering() {
        synchronized(lockTable) {
            List<Triple<NColumn, NStructureDataProvider,NColumn.NexusRowFilter>> filters = new ArrayList();
            for(Map.Entry<NColumn,List<NColumn.NexusRowFilter>> fiss : this.rowFilters.entrySet()) {
                for(NColumn.NexusRowFilter fi : fiss.getValue()) {
                    filters.add(Triple.of(fiss.getKey(), (NStructureDataProvider) getDatasetForColumn( fiss.getKey()),fi));
                }
            }
            // sort fastest filters first:
            filters.sort( (x,y) -> -Double.compare( x.getRight().getApproximateFilterSpeed() , y.getRight().getApproximateFilterSpeed() ) );

            BitSet in = new BitSet(this.allRows.size());
            in.set(0,this.allRows.size());
            for(int zi=0;zi<filters.size();zi++) {
                in.and( filters.get(zi).getRight().filterNexusRows( filters.get(zi).getMiddle() , this.allRows, in) );
            }
            this.visibleRows = new ArrayList<>();
            for(int zi=0;zi<this.allRows.size();zi++) {
                if(in.get(zi)) {visibleRows.add(this.allRows.get(zi));}
            }
            fireTableDataChanged();
        }
    }

    List<NexusTableModelListener> nexusListeners = new ArrayList<>();

    public void addNexusListener(NexusTableModelListener li) {
        nexusListeners.add(li);}
    public void removeNexusListener(NexusTableModelListener li) {
        nexusListeners.remove(li);}

    public void fireNexusTableStructureChangedEvent() {
        for(NexusTableModelListener li : this.nexusListeners) {
            SwingUtilities.invokeLater(new Runnable(){
                @Override
                public void run() {
                    li.nexusTableStructureChanged();
                }
            });
        }
    }

    public List<Pair<NColumn, NStructureDataProvider>> getNexusColumnsWithDataProviders() {
        List<Pair<NColumn, NStructureDataProvider>> colwdp = new ArrayList<>();
        for(NColumn ci : this.columns) {
            colwdp.add(Pair.of(ci, (NStructureDataProvider) this.getDatasetForColumn(ci)));
        }
        return colwdp;
    }

    public static interface NexusTableModelListener {
        public void nexusTableStructureChanged();
    }

}
