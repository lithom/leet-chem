package tech.molecules.leet.table;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.clustering.ClusterAppModel;

import javax.swing.*;
import javax.swing.border.LineBorder;
import javax.swing.event.CellEditorListener;
import javax.swing.table.TableCellEditor;
import javax.swing.table.TableCellRenderer;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.*;
import java.util.List;

public class ClassificationColumn implements NColumn<NDataProvider.StructureDataProvider, List<NClassification.NClass>> {

    private NClassification classification;

    public ClassificationColumn(NClassification classification) {
        this.classification = classification;

        classification.addClassificationListener(new NClassification.ClassificationListener() {
            @Override
            public void classificationChanged() {
                // TODO: fire column changed..
            }

            @Override
            public void classChanged(NClassification.NClass ci) {

            }
        });
    }

    @Override
    public String getName() {
        return "Classification";
    }


    private NexusTableModel ntm;

    @Override
    public void startAsyncInitialization(NexusTableModel model, NDataProvider.StructureDataProvider dataprovider) {
        this.ntm = model;
    }


    private List<CellSpecificAction> cellPopupActions = new ArrayList<>();

    @Override
    public void addCellPopupAction(CellSpecificAction ca) {
        this.cellPopupActions.add(ca);
    }

    @Override
    public List<NClassification.NClass> getData(NDataProvider.StructureDataProvider data, String rowid) {
        List<NClassification.NClass> classes = new ArrayList<>();
        return classes;
    }

    @Override
    public TableCellEditor getCellEditor() {
        return new ClassificationRenderer();
    }

    @Override
    public Map<String, NumericalDatasource<NDataProvider.StructureDataProvider>> getNumericalDataSources() {
        return new HashMap<>();
    }

    @Override
    public double evaluateNumericalDataSource(NDataProvider.StructureDataProvider dp, String datasource, String rowid) {
        return 0;
    }

    @Override
    public List<String> getRowFilterTypes() {
        return null;
    }

    @Override
    public NexusRowFilter<NDataProvider.StructureDataProvider> createRowFilter(NexusTableModel tableModel, String name) {
        return null;
    }


    @Override
    public void addColumnDataListener(ColumnDataListener cdl) {

    }

    @Override
    public boolean removeColumnDataListener(ColumnDataListener cdl) {
        return false;
    }


    public class ClassificationRenderer extends JPanel implements TableCellEditor , TableCellRenderer {


        public Component getTableCellRendererComponent (JTable table,
                                                        Object value,
                                                        boolean isSelected,
                                                        boolean isFocus,
                                                        int row,
                                                        int column)
        {
            // As a safety check, it's always good to verify the type of
            // value.

            if(value == null) {
                this.removeAll();
            }

            if (value instanceof List)
            {
                List<NClassification.NClass> classes = new ArrayList<>();
                ((List)value).stream().filter( ci -> ci instanceof NClassification.NClass).forEach( ci -> classes.add((NClassification.NClass)ci) );

                this.removeAll();
                this.setLayout( new GridLayout(classes.size(),1) );

                for(int zi=0;zi<classes.size();zi++) {
                    JColorLabel cla = new JColorLabel(classes.get(zi).getName(),classes.get(zi).getColor());
                    this.add(cla);
                }

                MouseAdapter mli = new MouseAdapter() {
                    @Override
                    public void mouseEntered(MouseEvent e) {
                        if( e.getComponent() instanceof JComponent) {
                            ((JComponent)e.getComponent()).setBorder(new LineBorder(Color.blue,1));
                        }
                    }
                    @Override
                    public void mouseExited(MouseEvent e) {
                        ((JComponent)e.getComponent()).setBorder(null);
                    }
                };
                //pa.addMouseListener(mli);
                //pb.addMouseListener(mli);
                //pc.addMouseListener(mli);
                //pd.addMouseListener(mli);

//            if (s.equals ("yellow"))
//                setBackground (Color.yellow);
//            else
//                setBackground (Color.white);
//            // Ensure text is displayed horizontally.
//            setHorizontalAlignment (CENTER);
//            // Assign the text to the JLabel component.
//            setText (s);
            }
            // Return a reference to the ColorRenderer (JLabel subclass)
            // component. Behind the scenes, that component will be used to
            // paint the pixels.
            return this;
        }

        private Object lastValue = null;
        @Override
        public Component getTableCellEditorComponent(JTable table, Object value, boolean isSelected, int row, int column) {
            this.lastValue = value;
            Component ci = getTableCellRendererComponent(table,value,isSelected,true,row,column);
            if(ci instanceof JComponent) {
                JPopupMenu pop = new JPopupMenu();
                String ridi =  ntm.getRowIdForVisibleRow(row);
                for(CellSpecificAction csi : cellPopupActions) {
                    csi.setRowId(ridi);
                    pop.add(csi);
                }
                //pop.add(new ClusterAppModel.CreateClusterAction("Create Cluster",()-> Pair.of("cx",Color.blue),()->Collections.singletonList(ridi) ));
                ((JComponent)ci).setComponentPopupMenu(pop);
            }
            return ci;
        }

        @Override
        public Object getCellEditorValue() {
            return this.lastValue;
        }

        @Override
        public boolean isCellEditable(EventObject anEvent) {
            return true;
        }

        @Override
        public boolean shouldSelectCell(EventObject anEvent) {
            return true;
//            if( anEvent instanceof MouseEvent) {
//                MouseEvent me = (MouseEvent) anEvent;
//                return true;
//            }
//            return false;
        }



        @Override
        public boolean stopCellEditing() {
            return false;
        }

        @Override
        public void cancelCellEditing() {

        }

        @Override
        public void addCellEditorListener(CellEditorListener l) {

        }

        @Override
        public void removeCellEditorListener(CellEditorListener l) {

        }
    }

    public static class JColorLabel extends JLabel {
        public JColorLabel(String text, Color ca) {
            super(text);
            Color c = new Color( ca.getRed() , ca.getGreen() , ca.getBlue() , 40 );
            this.setOpaque(true);
            this.setBackground(c);
        }
    }


}
