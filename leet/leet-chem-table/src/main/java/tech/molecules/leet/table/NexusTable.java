package tech.molecules.leet.table;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.table.gui.JFilterPanel;
import tech.molecules.leet.table.gui.JNumericalDataSourceSelector;
import tech.molecules.leet.util.ColorMapHelper;

import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.table.TableCellEditor;
import javax.swing.table.TableCellRenderer;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

public class NexusTable extends JTable {

    private NexusTableModel model;

    private int mouseOverRow = -1;
    private int mouseOverCol = -1;
    private String mouseOverKey = null;

    private Set<String> currentSelection = new HashSet<>();

    private NexusTableModel.SelectionType selectionTypeSelected;
    private NexusTableModel.SelectionType selectionTypeMouseOver;

    public NexusTable(NexusTableModel model) {
        super(model);
        //this.model = model;
        //this.updateColumns();

        this.selectionTypeSelected  = model.getSelectionType(NexusTableModel.SELECTION_TYPE_SELECTED);
        this.selectionTypeMouseOver = model.getSelectionType(NexusTableModel.SELECTION_TYPE_MOUSE_OVER);

        this.setModel(model);
        this.topPanel = new TopPanel();
        this.topPanel.setSliderManually(60);

        this.getSelectionModel().addListSelectionListener(new ListSelectionListener() {
            @Override
            public void valueChanged(ListSelectionEvent e) {
                Set<String> newCurrentSelection = new HashSet<>();
                model.removeSelectionTypeFromRows(model.getSelectionType(NexusTableModel.SELECTION_TYPE_SELECTED), currentSelection);
                for(int ri : getSelectedRows()) {
                    String kri = model.getKeyAtVisiblePosition(ri);
                    newCurrentSelection.add(kri);
                }
                model.addSelectionTypeToRows(selectionTypeSelected, Arrays.stream(getSelectedRows()).mapToObj( ri -> model.getKeyAtVisiblePosition(ri) ).collect(Collectors.toList()) );
                currentSelection = newCurrentSelection;
            }
        });

        this.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseExited(MouseEvent e) {
                mouseOverRow = -1;
                mouseOverCol = -1;
                repaint();
            }
        });
        this.addMouseMotionListener(new MouseMotionListener() {
            @Override
            public void mouseDragged(MouseEvent e) {

            }

            @Override
            public void mouseMoved(MouseEvent e) {
                int ci = getThisTable().columnAtPoint(e.getPoint());
                int ri = getThisTable().rowAtPoint(e.getPoint());
                System.out.println("mm: "+ri+"/"+ci);

                // row changed? (row highlight)
                if(ci>=0&&ri>=0) {
                    String ki = model.getKeyAtVisiblePosition(ri);
                    if(ki!=null && !ki.equals(mouseOverKey)) {
                        // row changed, set new highlight
                        //if(mouseOverRow>=0) {
                        if(mouseOverKey!=null) {
                            model.removeSelectionTypeFromRows(selectionTypeMouseOver,Collections.singletonList(mouseOverKey));
                        }
                        model.addSelectionTypeToRows(selectionTypeMouseOver,Collections.singletonList(ki));
                        mouseOverKey = ki;
                    }
                }

                if(ci>=0&&ri>=0) {
                    if(mouseOverCol!=ci || mouseOverRow!=ri) {
                        mouseOverCol = ci;
                        mouseOverRow = ri;
                        System.out.println("set edit cell: "+ri+"/"+ci);
                        editCellAt(ri,ci);
                        repaint();
                    }
                }
                else {
                    if(mouseOverCol!=ci || mouseOverRow!=ri) {
                        //model.removeSelectionColorFromRow(model.getKeyAtVisiblePosition(mouseOverRow),colorMouseOverRow);
                        model.removeSelectionTypeFromRows(selectionTypeMouseOver,Collections.singletonList(model.getKeyAtVisiblePosition(mouseOverRow)));
                        mouseOverCol = -1;
                        mouseOverRow = -1;
                        editCellAt(ri,ci);
                        repaint();
                    }
                }
            }
        });

    }

    public void paintComponent(Graphics g) {
        super.paintComponent(g);

        if(false) {
            if (mouseOverCol >= 0 && mouseOverRow >= 0) {
                Rectangle r = getCellRect(mouseOverRow, mouseOverCol, true);
                Rectangle.Double r_col = new Rectangle.Double(r.getX(), 0, r.getWidth(), getHeight());
                Rectangle.Double r_row = new Rectangle.Double(0, r.getY(), getWidth(), r.getHeight());
                Graphics2D g2 = (Graphics2D) g;
                g2.setPaint(new Color(240, 165, 20, 20));
                g2.fill(r_col);
                g2.fill(r_row);
                g2.setPaint(Color.red);
                g2.draw(r);
            }
        }
    }

    public NexusTable getThisTable() {
        return this;
    }

    public NexusTableModel getTableModel() {
        return this.model;
    }

    private MouseListener headerMouseListener = null;

    public void updateColumns() {
        //List<Pair<NColumn, NStructureDataProvider>> ncolumns = this.getTableModel().getNexusColumnsWithDataProviders();
        List<Pair<NColumn, NDataProvider>> ncolumns = this.getTableModel().getNexusColumnsWithDataProviders();
        for( int zi=0;zi<ncolumns.size();zi++) {
            if(ncolumns.get(zi).getLeft().getCellEditor()!=null) {
                this.getColumnModel().getColumn(zi).setCellEditor(ncolumns.get(zi).getLeft().getCellEditor());
                this.getColumnModel().getColumn(zi).setCellRenderer( new DefaultRendererFromEditor (ncolumns.get(zi).getLeft().getCellEditor()));
            }
            //this.getColumnModel().getColumn(zi).setCellRenderer(ncolumns.get(zi).getLeft().getCellRenderer());
            this.getColumnModel().getColumn(zi).setHeaderValue(ncolumns.get(zi).getLeft().getName());
        }

        // update listener for column header
        this.getTableHeader().removeMouseListener(headerMouseListener);
        this.headerMouseListener = new MouseAdapter() {
            private void maybeShowPopup(MouseEvent e) {
                if(e.isPopupTrigger()) {
                    int column = columnAtPoint(e.getPoint());
                    //NColumn col = model.getNexusColumns().get(column).getLeft();
                    NColumn col = ncolumns.get(column).getLeft();
                    JPopupMenu popup = new JPopupMenu();
                    //popup.add(new JMenuItem("gugus! from "+cname));
                    // add filtering menu:
                    JMenu filtering = new JMenu("Add Filter");
                    //for(Object filtertype : model.getNexusColumns().get(column).getLeft().getRowFilterTypes()) {
                    for(Object filtertype : ncolumns.get(column).getLeft().getRowFilterTypes()) {
                        JMenuItem ji = new JMenuItem( (String) filtertype );
                        filtering.add(ji);
                        ji.addActionListener(new ActionListener() {
                            @Override
                            public void actionPerformed(ActionEvent e) {
                                //NColumn.NexusRowFilter filter = model.getNexusColumns().get(column).getLeft().createRowFilter( model , (String) filtertype );
                                NColumn.NexusRowFilter filter = ncolumns.get(column).getLeft().createRowFilter( model , (String) filtertype );
                                filter.setupFilter(model,ncolumns.get(column).getRight());
                                model.addRowFilter(col,filter);
                                filterPanel.addFilter(filter);
                                //filterPanel.add(filter.getFilterGUI());
                                filterPanel.invalidate();
                                invalidate();
                                filterPanel.repaint();
                            }
                        });
                    }
                    popup.add(filtering);
                    popup.show(e.getComponent(),e.getX(),e.getY());
                }
            }

            @Override
            public void mousePressed(MouseEvent e) {
                maybeShowPopup(e);
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                maybeShowPopup(e);
            }
        };
        this.getTableHeader().addMouseListener(this.headerMouseListener);
    }

    public class TopPanel extends JPanel {
        private JMenuBar jmb = new JMenuBar();
        private JSlider slider = new JSlider(16,320);
        public TopPanel() {
            this.setLayout(new FlowLayout());
            this.add(new JLabel("RowHeight "));
            this.add(slider);
            this.add(jmb);
            slider.addChangeListener(e -> setRowHeight(slider.getValue()));
            reinitMenu();
            model.addNexusListener(new NexusTableModel.NexusTableModelListener() {
                @Override
                public void nexusTableStructureChanged() {
                    reinitMenu();
                }
            });
        }
        public void setSliderManually(int size) {
            slider.setValue(size);
        }
        private void reinitMenu() {
            jmb.removeAll();
            JMenu jm_color = new JMenu("Coloring");
            if(getTableModel()!=null) {
                JNumericalDataSourceSelector jndss = new JNumericalDataSourceSelector(new JNumericalDataSourceSelector.NumericalDataSourceSelectorModel(getTableModel()), JNumericalDataSourceSelector.SELECTOR_MODE.Menu);
                jm_color.add(jndss.getMenu());
                jndss.addSelectionListener(new JNumericalDataSourceSelector.SelectionListener() {
                    @Override
                    public void selectionChanged() {
                        // set coloring according to values of nds
                        model.setHighlightColors(ColorMapHelper.evaluateColorValues(null, getTableModel(), jndss.getModel().getSelectedDatasource()));
                    }
                });
            }
            jmb.add(jm_color);
        }
    }

    private TopPanel topPanel;

    public TopPanel getTopPanel() {
        return topPanel;
    }

    public void setModel(NexusTableModel model) {
        super.setModel(model);
        this.model = model;
        this.updateColumns();
        this.getTableModel().addNexusListener(new NexusTableModel.NexusTableModelListener() {
            @Override
            public void nexusTableStructureChanged() {
                updateColumns();
            }
        });
    }

    private JFilterPanel filterPanel = new JFilterPanel();

    public JFilterPanel getFilterPanel() {
        return filterPanel;
    }

    public static class DefaultRendererFromEditor implements TableCellRenderer {
        private TableCellEditor editor;
        public DefaultRendererFromEditor(TableCellEditor editor) {
            this.editor = editor;
        }
        @Override
        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
            return this.editor.getTableCellEditorComponent(table,value,isSelected,row,column);
        }
    }


    private Color colorMouseOverRow = Color.cyan.darker();
    private Color colorListSelection = Color.orange.darker().darker();

    public void setMouseOverHighlightingEventColor(boolean useMouseOverHighlighting, Color col) {
        if(useMouseOverHighlighting) {
            colorMouseOverRow = col;
        }
        else {
            colorMouseOverRow = null;
        }
    }
    public void setListSelectionColor(Color col) {
        this.colorListSelection = col;
    }







    public static class DefaultEditorFromRenderer extends AbstractCellEditor implements TableCellEditor {
        TableCellRenderer ra;
        public DefaultEditorFromRenderer(TableCellRenderer ra) {
            this.ra = ra;
        }
        Object lastValue = null;
        @Override
        public Component getTableCellEditorComponent(JTable table, Object value, boolean isSelected, int row, int column) {
            lastValue = value;
            return this.ra.getTableCellRendererComponent(table,value,isSelected,false,row,column);
        }

        @Override
        public Object getCellEditorValue() {
            return lastValue;
        }
    }

    public static JCellBackgroundPanel getDefaultEditorBackgroundPanel(NexusTable nt, NexusTableModel.NexusHighlightingAndSelectionStatus status) {
        return new JCellBackgroundPanel(nt,status);
    }

    public static class JCellBackgroundPanel extends JPanel {
        private int SELECTION_LINE_SIZE   = 6;
        private int SELECTION_LINE_LENGTH = 20;
        private NexusTable nt;
        private NexusTableModel.NexusHighlightingAndSelectionStatus status;
        public JCellBackgroundPanel(NexusTable nt, NexusTableModel.NexusHighlightingAndSelectionStatus status) {
            this.nt = nt;
            this.status = status;
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g;
            if(this.status.highlightingColor!=null) {
                Color cia = new Color(status.highlightingColor.getRed(), status.highlightingColor.getGreen(), status.highlightingColor.getBlue(), 80);
                Color cib = new Color(status.highlightingColor.getRed(), status.highlightingColor.getGreen(), status.highlightingColor.getBlue(), 60);
                GradientPaint gpa = new GradientPaint(0, 0, cia, 0, getHeight(), cib);
                g2.setPaint(gpa);
                g2.fillRect(0, 0, getWidth(), getHeight());
            }
            if(this.status.selectionColor!=null && this.status.selectionColor.size()>0) {
                g2.setStroke(new BasicStroke(SELECTION_LINE_SIZE));
                if(true) {
                    if(status.selectionColor.size()==1) {
                        Color cia = new Color(status.selectionColor.get(0).getColor().getRed(), status.selectionColor.get(0).getColor().getGreen(), status.selectionColor.get(0).getColor().getBlue(), 140);
                        g2.setColor(cia);
                        g2.drawLine(0,0,this.getWidth(),0);
                        g2.drawLine(0,this.getHeight(),this.getWidth(),this.getHeight());
                        g2.drawLine(0,0,0,this.getHeight());
                        g2.drawLine(this.getWidth(),0,this.getWidth(),this.getHeight());
                    }
                    else {
                        for (int zi = 0; (zi) * SELECTION_LINE_LENGTH < this.getWidth(); zi++) {
                            int xa = SELECTION_LINE_LENGTH * zi;
                            int xb = Math.min(this.getWidth(), xa + SELECTION_LINE_LENGTH);
                            int cxi = zi % this.status.selectionColor.size();
                            Color cia = new Color(status.selectionColor.get(cxi).getColor().getRed(), status.selectionColor.get(cxi).getColor().getGreen(), status.selectionColor.get(cxi).getColor().getBlue(), 140);
                            g2.setColor(cia);
                            // draw top and bottom line
                            g2.drawLine(xa, 0, xb, 0);
                            g2.drawLine(xa, this.getHeight(), xb, this.getHeight());
                        }
                        for (int zi = 0; (zi) * SELECTION_LINE_LENGTH < this.getHeight(); zi++) {
                            int ya = SELECTION_LINE_LENGTH * zi;
                            int yb = Math.min(this.getHeight(), ya + SELECTION_LINE_LENGTH);
                            int cxi = zi % this.status.selectionColor.size();
                            Color cia = new Color(status.selectionColor.get(cxi).getColor().getRed(), status.selectionColor.get(cxi).getColor().getGreen(), status.selectionColor.get(cxi).getColor().getBlue(), 140);
                            g2.setColor(cia);
                            // draw top and bottom line
                            g2.drawLine(0, ya, 0, yb);
                            g2.drawLine(this.getWidth(), ya, this.getWidth(), yb);
                        }
                    }
                }
                if(false) {
                    for (int zi = 0; zi < this.status.selectionColor.size(); zi++) {
                        Color cia = new Color(status.selectionColor.get(zi).getColor().getRed(), status.selectionColor.get(zi).getColor().getGreen(), status.selectionColor.get(zi).getColor().getBlue(), 180);
                        g2.setColor(cia);
                        // draw top and bottom line
                        int cy = (int) ((zi) * (SELECTION_LINE_SIZE * 1.0));
                        g2.drawLine(0, cy, this.getWidth(), cy);
                        g2.drawLine(0, this.getHeight() - cy, this.getWidth(), this.getHeight() - cy);
                    }
                }
            }
        }
    }

}
