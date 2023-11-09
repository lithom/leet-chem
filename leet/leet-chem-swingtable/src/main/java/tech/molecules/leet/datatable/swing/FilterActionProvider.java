package tech.molecules.leet.datatable.swing;

import tech.molecules.leet.datatable.DataFilter;
import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.DataTableColumn;

import javax.swing.*;


/**
 * Resolves Filter Actions based on Filters that exist in the DataTable.
 * I.e. instances of this provide / create the GUIs for filters.
 *
 */
public interface FilterActionProvider {
    public Action getAddFilterAction(DataTable dtable, DataTableColumn dcol, DataFilter filter);
}
