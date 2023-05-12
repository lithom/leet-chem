package tech.molecules.leet.datatable;

import javax.swing.table.AbstractTableModel;
import java.util.List;

public class SwingTableModelAdapter extends AbstractTableModel {

    private DataTable dataTable;

    public SwingTableModelAdapter(DataTable dataTable) {
        this.dataTable = dataTable;

        initReloading();
        dataTable.addDataTableListener(new DataTable.DataTableListener() {
            @Override
            public void tableStructureChanged() {
                fireTableStructureChanged();
            }

            @Override
            public void tableDataChanged() {
                fireTableDataChanged();
            }

            @Override
            public void tableCellsChanged(List<int[]> cells) {
                // TODO implement table cells changed event forwarding
                for(int[] ci : cells) {
                    fireTableCellUpdated(ci[0],ci[1]);
                }
            }
        });
    }

    private void initReloading() {
        //dataTable.getDataColumns()
        fireTableStructureChanged();
        fireTableDataChanged();
    }

    @Override
    public int getRowCount() {
        return dataTable.getVisibleKeysSorted().size();
    }

    @Override
    public int getColumnCount() {
        return dataTable.getDataColumns().size();
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        return dataTable.getValue(rowIndex,columnIndex);
        //return dataTable.getDataColumns().get(columnIndex).getValue( dataTable.getVisibleKeysSorted().get(rowIndex) );
    }
}
