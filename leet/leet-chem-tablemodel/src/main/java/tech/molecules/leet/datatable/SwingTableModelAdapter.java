package tech.molecules.leet.datatable;

import javax.swing.table.AbstractTableModel;

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
        });
    }

    private void initReloading() {
        //dataTable.getDataColumns()
        fireTableStructureChanged();
        fireTableDataChanged();
    }

    @Override
    public int getRowCount() {
        return dataTable.getVisibleKeys().size();
    }

    @Override
    public int getColumnCount() {
        return dataTable.getDataColumns().size();
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        return dataTable.getDataColumns().get(columnIndex).getValue( dataTable.getVisibleKeys().get(rowIndex) );
    }
}
