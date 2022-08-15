package tech.molecules.leet.table;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.table.gui.JFilterPanel;

import javax.swing.*;
import javax.swing.table.TableCellEditor;
import javax.swing.table.TableCellRenderer;
import java.awt.*;
import java.awt.event.*;
import java.util.List;

public class NexusTable extends JTable {

    private NexusTableModel model;

    private int mouseOverRow = -1;
    private int mouseOverCol = -1;



    public NexusTable(NexusTableModel model) {
        super(model);
        //this.model = model;
        //this.updateColumns();
        this.setModel(model);
        this.topPanel.setSliderManually(60);



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
                if(ci>=0&&ri>=0) {
                    if(mouseOverCol!=ci || mouseOverRow!=ri) {
                        mouseOverCol = ci;
                        mouseOverRow = ri;
                        editCellAt(ri,ci);
                        repaint();
                    }
                }
                else {
                    if(mouseOverCol!=ci || mouseOverRow!=ri) {
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
        private JSlider slider = new JSlider(16,320);
        public TopPanel() {
            this.setLayout(new FlowLayout());
            this.add(slider);
            slider.addChangeListener(e -> setRowHeight(slider.getValue()));
        }
        public void setSliderManually(int size) {
            slider.setValue(size);
        }
    }

    private TopPanel topPanel = new TopPanel();

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

}
