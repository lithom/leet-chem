package tech.molecules.leet.datatable.swing;

import tech.molecules.leet.datatable.DataTableColumn;

import javax.swing.table.TableCellRenderer;

public class DefaultSwingDataTableColumnController<T,U> {

    private DataTableColumn<T,U> column;
    private TableCellRenderer renderer;

    public DefaultSwingDataTableColumnController(DataTableColumn<T, U> column, TableCellRenderer renderer) {
        this.column = column;
        this.renderer = renderer;
    }

    public DataTableColumn<T, U> getColumn() {
        return column;
    }

    public void setColumn(DataTableColumn<T, U> column) {
        this.column = column;
    }

    public TableCellRenderer getRenderer() {
        return this.renderer;
    }

    public void setRenderer(TableCellRenderer renderer) {
        this.renderer = renderer;
    }
}
