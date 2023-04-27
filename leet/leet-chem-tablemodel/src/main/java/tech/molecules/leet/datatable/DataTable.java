package tech.molecules.leet.datatable;

import java.awt.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 *
 * Event Handling: Data columns have their own data provider that always
 * "immediately" provides them with data. However, data columns should not
 * themselves listen for changed in the data provider. Instead, data providers
 * can be registered in the DataTable, and then in case that a data provider
 * sends a data changed event, it notifies repaint events to all columns that
 * use the given data provider.
 *
 *
 */
public class DataTable {

    public interface DataTableTask<T> {
        void process();
    }

    private BlockingQueue<DataTableTask> taskQueue;
    private Thread processingThread;

    public void initDataTableUpdateThread() {
        taskQueue = new LinkedBlockingQueue<>();
        processingThread = new Thread(() -> {
            while (true) {
                try {
                    DataTableTask task = taskQueue.take();
                    task.process();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        });
        processingThread.start();
    }



    private List<DataTableColumn> columns;


    public List<DataTableColumn> getDataColumns() {
        synchronized (this.columns) {
            return new ArrayList<>(this.columns);
        }
    }

    public void addDataColumn(DataTableColumn dtc) {
        synchronized (this.columns) {
            this.columns.add(dtc);
            this.filters.put(dtc,new ArrayList<>());
        }
    }

    public boolean removeDataColumn(DataTableColumn dtc) {
        synchronized (this.columns) {
            return this.columns.add(dtc);
        }
    }


    private List<String> visibleKeys;

    public List<String> getVisibleKeys() {
        synchronized (visibleKeys) {
            return new ArrayList<>(visibleKeys);
        }
    }

    private void setVisibleKeys(List<String> vk) {
        synchronized(visibleKeys) { this.visibleKeys = vk; }
    }


    private Map<DataTableColumn,List<AbstractDataFilter>> filters;

    public boolean removeFilter(DataTableColumn dtc, AbstractDataFilter fi) {
        boolean removed = false;
        synchronized(this.columns) {
            removed = this.filters.get(dtc).remove(fi);
        }
        if(removed) {
            this.taskQueue.add(new UpdateRowFilteringTask());
        }
        return removed;
    }

    public void addFilter(DataTableColumn dtc, AbstractDataFilter fi) {
        synchronized(this.columns) {
            this.filters.get(dtc).add(fi);
        }
        this.taskQueue.add(new UpdateRowFilteringTask());
    }

    private class UpdateRowFilteringTask implements DataTableTask {
        @Override
        public void process() {

        }
    }

    private List<DataTableListener> listeners = new ArrayList<>();

    public void addDataTableListener(DataTableListener li) {
        listeners.add(li);
    }

    public boolean removeDataTableListener(DataTableListener li) {
        return listeners.remove(li);
    }


    public static class CellState {
        public final Color backgroundColor;
        public final List<Color> selectionColors;
        public CellState(Color backgroundColor, List<Color> selectionColors) {
            this.backgroundColor = backgroundColor;
            this.selectionColors = selectionColors;
        }
    }

    public DataTable.CellState getCellState(int row, int col) {
        return new CellState(null, Arrays.asList(new Color[]{Color.red,Color.blue}));
    }


    private void fireTableStructureChanged() {
        for(DataTableListener li : listeners) {
            li.tableStructureChanged();
        }
    }

    private void fireTableCellChanged(List<int[]> cells) {
        for(DataTableListener li : listeners) {
            li.tableCellsChanged(cells);
        }
    }

    private void fireTableDataChanged() {
        for(DataTableListener li : listeners) {
            li.tableDataChanged();
        }
    }

    public static interface DataTableListener {
        public void tableStructureChanged();
        public void tableDataChanged();
        /**
         *
         * @param cells entries are {row,col}
         */
        public void tableCellsChanged(List<int[]> cells);
    }

}
