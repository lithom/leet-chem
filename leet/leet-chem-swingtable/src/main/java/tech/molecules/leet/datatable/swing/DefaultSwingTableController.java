package tech.molecules.leet.datatable.swing;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.datatable.*;
import tech.molecules.leet.datatable.microleet.MicroLeetMain;
import tech.molecules.leet.gui.UtilSwing;

import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.TableCellRenderer;
import javax.swing.table.TableColumn;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.TreeModel;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Collectors;

public class DefaultSwingTableController extends JPanel {

    private DefaultSwingTableModel model;
    private List<FilterActionProvider> filterActionProviders;

    private Supplier<JPanel> filterPanelSupplier;

    private JTable table = new JTable();
    private JScrollPane scrollPane = new JScrollPane(table);


    public DefaultSwingTableController(DefaultSwingTableModel model) {
        this(model, new ArrayList<>(), new UtilSwing.PanelAsFrameProvider(null, 240,240) );
    }

    public DefaultSwingTableController(DefaultSwingTableModel model,
                                       List<FilterActionProvider> filterActionProviders,
                                       Supplier<JPanel> filterPanelSupplier) {
        this.model = model;
        this.filterActionProviders = filterActionProviders;
        this.filterPanelSupplier = filterPanelSupplier;

        reinitGUI();
        reinitController();
    }

    private void reinitGUI() {
        this.removeAll();
        this.setLayout(new BorderLayout());

        this.scrollPane = new JScrollPane();
        this.scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
        this.scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
        this.table      = new JTable();
        this.table.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
        this.scrollPane.setViewportView(this.table);
        this.add(this.scrollPane,BorderLayout.CENTER);
    }

    //private Map<Integer, List<Action>> setRendererActions = new HashMap<>();
    //private Map<Integer, List<Action>> addFilterActions = new HashMap<>();

    // Controller Listeners!
    private ListSelectionListener listSelectionListener = new ListSelectionListener() {
        @Override
        public void valueChanged(ListSelectionEvent e) {
            DataTableSelectionModel.SelectionType st_selected = model.getDataTable().getSelectionModel().getSelectionType(DataTableSelectionModel.SELECTION_TYPE_SELECTED);
            model.getDataTable().getSelectionModel().resetSelectionForSelectionType(st_selected);
            model.getDataTable().getSelectionModel().addSelectionTypeToRows(st_selected, model.getDataTable().getVisibleKeysSortedAt(table.getSelectedRows()));
        }
    };

    private DataTable.DataTableListener dataTableListener = new DataTable.DataTableListener() {
        @Override
        public void tableDataChanged() {

        }

        @Override
        public void tableStructureChanged() {
            reinitController();
        }

        @Override
        public void tableCellsChanged(List<int[]> cells) {

        }
    };


    private boolean mouseOverTable = false;
    private int mouseOverCol = -1;
    private int mouseOverRow = -1;

    private MouseListener tablePositionAndClickListener = new MouseAdapter() {
        @Override
        public void mouseClicked(MouseEvent e) {
            if (SwingUtilities.isRightMouseButton(e)) {
                //int column = table.columnAtPoint(e.getPoint());
                JPopupMenu popupMenu = createMainPopupMenu();
                popupMenu.show(table, e.getX(), e.getY());
            }
        }
        @Override
        public void mouseEntered(MouseEvent e) {
            mouseOverTable = true;
            mouseOverCol = -1;
            mouseOverRow = -1;
        }
        @Override
        public void mouseExited(MouseEvent e) {
            mouseOverTable = false;
        }
        @Override
        public void mouseMoved(MouseEvent e) {
            // Mouse moved over the table
            int colM = table.columnAtPoint(e.getPoint());
            int rowM    = table.rowAtPoint(e.getPoint());
            //System.out.println("Mouse over column: " + column);
            mouseOverCol = colM;
            mouseOverRow = rowM;
        }
    };

    private void reinitController() {
        this.table.setModel(this.model.getSwingTableModel());

        // TODO: init Selection Model listener
        this.table.getSelectionModel().removeListSelectionListener(this.listSelectionListener);
        this.table.getSelectionModel().addListSelectionListener(this.listSelectionListener);

        // listen to DataTable when table structure changes:
        this.model.getDataTable().removeDataTableListener(dataTableListener);
        this.model.getDataTable().addDataTableListener(dataTableListener);


        // install mouse listener to determine mouse over state:
        this.table.removeMouseListener(tablePositionAndClickListener);
        this.table.removeMouseMotionListener( (MouseMotionListener) tablePositionAndClickListener);
        this.table.addMouseListener(tablePositionAndClickListener);
        this.table.addMouseMotionListener( (MouseMotionListener) tablePositionAndClickListener);

        // TODO: install all column specific mouse listeners
        // i.e. this includes setting the renderer,
        // + filtering / sorting options
        // + alternative representations options
        // + numeric datasource options

        // reconfigureComponentPopupMenu();





        //this.table.setComponentPopupMenu(pop);


        // Layout / Table config stuff
        this.ensureMinColWidth(120);
    }

    private JPopupMenu createMainPopupMenu() {
        JPopupMenu pop = new JPopupMenu();

        // column specific:
        if(mouseOverTable) {
            DataTableColumn ci = this.getModel().getDataTable().getDataColumns().get(mouseOverCol);
            JMenu col_i_filter = new JMenu("Filters");
            createColumnSpecificFilterActions(ci).forEach( xi ->col_i_filter.add(xi) );
            pop.add(col_i_filter);
        }


        // Stuff for all columns:
        JMenu all_columns = new JMenu("All columns - Filter");
        for(DataTableColumn ci : this.getModel().getDataTable().getDataColumns()) {
            JMenu all_col_i = new JMenu(ci.toString());
            //fa.stream().forEach( xi -> pop.add(xi.getRight()) );
            createColumnSpecificFilterActions(ci).forEach( xi -> all_col_i.add(xi) );
            all_columns.add(all_col_i);
        }
        pop.add(all_columns);

        return pop;
    }

    private List<Action> createColumnSpecificFilterActions(DataTableColumn ci) {

        List<DataFilter> filters = ci.getFilters();
        List<Action> colFilterActions = new ArrayList<>();
        // try to create add filter actions:
        for(DataFilter fi : filters) {
            for(FilterActionProvider fpi : this.filterActionProviders) {
                Action xi = fpi.getAddFilterAction(getModel().getDataTable() ,ci, fi);
                if(xi != null) {
                    colFilterActions.add(xi);
                }
            }
        }

        //List<Pair<String[],Action>> fa = NumericDatasourceHelper.createFilterActions(this.getModel().getDataTable(),ci, new UtilSwing.PanelAsFrameProvider(this,240,240) );
        List<Pair<String[],Action>> fa = NumericDatasourceHelper.createFilterActions(this.getModel().getDataTable(),ci,filterPanelSupplier);
        List<Action> numericDatasourceFilterActions = fa.stream().map(xi -> xi.getRight()).collect(Collectors.toList());

        List<Action> all = new ArrayList<>();
        all.addAll(colFilterActions);
        all.addAll(numericDatasourceFilterActions);
        return all;
    }
    public void ensureMinColWidth(int minWidth) {
        int total_width = 0;
        for (int columnIndex = 0; columnIndex < table.getColumnCount(); columnIndex++) {
            TableColumn column = table.getColumnModel().getColumn(columnIndex);
            int preferredWidth = Math.max(column.getWidth(), minWidth);
            total_width += preferredWidth;
            column.setPreferredWidth(preferredWidth);
        }
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
            //JPanel panel = new JPanel();
            //panel.setBackground(cellState.backgroundColor);
            if(cellState.backgroundColor!=null) {
                component.setBackground(cellState.backgroundColor);
            }

            // Create a custom border with the selection colors
            //panel.setBorder(new MultiColorBorder(cellState.selectionColors, 40));
            if(component instanceof JComponent) {
                ((JComponent)component).setBorder(new MultiColorBorder(cellState.selectionColors, 200));
            }

            return component;
            // Add the original component (from the specific renderer) to the panel
            //panel.add((JComponent) component);
            //return panel;
        }
    }

    public void setRowHeight(int rh) {
        this.table.setRowHeight(rh);
    }

    public DefaultSwingTableModel getModel() {
        return this.model;
    }

    public TreeModel createDatasourcesTreeModel() {
        DefaultMutableTreeNode root = new DefaultMutableTreeNode();
        for(DataTableColumn ci : this.model.getDataTable().getDataColumns()) {
            DefaultMutableTreeNode ni = new DefaultMutableTreeNode(ci);
            List<NumericDatasource> ndi = ci.getNumericDatasources();
            for(NumericDatasource nd : ndi) {
                ni.add( new DefaultMutableTreeNode(nd));
            }
            List<CategoricDatasource> cdi = ci.getCategoricDatasources();
            for(CategoricDatasource cd : cdi) {
                ni.add( new DefaultMutableTreeNode(cd));
            }
        }
        return new DefaultTreeModel(root);
    }

    public TreeModel createActiveFilterTreeModel() {
        DefaultMutableTreeNode root = new DefaultMutableTreeNode();
        Map<DataTableColumn,List<DataFilter>> filters = this.model.getDataTable().getFiltersSorted();
        for(DataTableColumn ci : filters.keySet()) {
            DefaultMutableTreeNode ni = new DefaultMutableTreeNode(ci);
            List<DataFilter> fix = filters.get(ci);
            for(DataFilter fi : fix) {
                ni.add( new DefaultMutableTreeNode(fi));
            }
            if(fix.size()>0) {
                root.add(ni);
            }
        }
        return new DefaultTreeModel(root);
    }


}
