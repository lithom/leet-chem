package tech.molecules.leet.datatable.swing;

import tech.molecules.leet.datatable.DataTable;

import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.TableCellRenderer;
import java.awt.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DefaultSwingTableController extends JPanel {

    private DefaultSwingTableModel model;

    private JTable table = new JTable();
    private JScrollPane scrollPane = new JScrollPane(table);


    public DefaultSwingTableController(DefaultSwingTableModel model) {
        this.model = model;
        reinitGUI();
        reinitController();
    }

    private void reinitGUI() {
        this.removeAll();
        this.setLayout(new BorderLayout());

        this.scrollPane = new JScrollPane();
        this.table      = new JTable();
        this.scrollPane.setViewportView(this.table);
        this.add(this.scrollPane,BorderLayout.CENTER);
    }

    //private Map<Integer, List<Action>> setRendererActions = new HashMap<>();
    //private Map<Integer, List<Action>> addFilterActions = new HashMap<>();

    private void reinitController() {
        this.table.setModel(this.model.getSwingTableModel());

        // TODO: install all column specific mouse listeners
        // i.e. this includes setting the renderer,
        // + filtering / sorting options
        // + alternative representations options
        // + numeric datasource options
    }

    public void setTableCellRenderer(int col, TableCellRenderer renderer) {
        this.table.getColumn(""+col).setCellRenderer(new DataTableCellRendererAdapter(this.model.getDataTable(),renderer));
    }

    public static class DataTableCellRendererAdapter extends DefaultTableCellRenderer {

        private DataTable dtm;
        private TableCellRenderer wrappedTableCellRenderer;

        public DataTableCellRendererAdapter(DataTable dtm, TableCellRenderer wrappedTableCellRenderer) {
            this.dtm = dtm;
            this.wrappedTableCellRenderer = wrappedTableCellRenderer;
        }

        @Override
        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected,
                                                       boolean hasFocus, int row, int column) {
            Component component = this.wrappedTableCellRenderer.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, column);

            // Retrieve the CellState for the current cell
            DataTable.CellState cellState = this.dtm.getCellState(row, column); // Implement this method to get the CellState from your data model

            // Create a JPanel with the specified background color
            JPanel panel = new JPanel();
            panel.setBackground(cellState.backgroundColor);

            // Create a custom border with the selection colors
            panel.setBorder(new MultiColorBorder(cellState.selectionColors, 40));

            // Add the original component (from the specific renderer) to the panel
            panel.add((JComponent) component);

            return panel;
        }
    }
}
