package tech.molecules.leet.table;

import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.clustering.ClusterAppModel;
import tech.molecules.leet.table.gui.DispatchingMouseAdapter;

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


// TODO: maybe we have to add mousePressed event dispatching to Nexus infrastructure..
public class ClassificationColumn implements NColumn<NDataProvider.ClassificationProvider, List<NClassification.NClass>> {

    private NDataProvider.ClassificationProvider dp;

    @Override
    public void setDataProvider(NDataProvider.ClassificationProvider dataprovider) {
        this.dp = dataprovider;
    }

    @Override
    public NDataProvider.ClassificationProvider getDataProvider() {
        return this.dp;
    }

    public ClassificationColumn(NClassification classification) {
        //this.classification = classification;

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
    public void startAsyncReinitialization(NexusTableModel model) {
        this.ntm = model;
    }


    private List<CellSpecificAction> cellPopupActions = new ArrayList<>();

    @Override
    public void addCellPopupAction(CellSpecificAction ca) {
        this.cellPopupActions.add(ca);
    }

    @Override
    public List<NClassification.NClass> getData(String rowid) {
        //List<NClassification.NClass> classes = new ArrayList<>();
        return this.dp.getClassification().getClassesForRow(rowid);
    }

    @Override
    public TableCellEditor getCellEditor() {
        return new ClassificationRenderer();
    }

    @Override
    public Map<String, NumericalDatasource<NDataProvider.ClassificationProvider>> getNumericalDataSources() {
        return new HashMap<>();
    }

    @Override
    public List<String> getRowFilterTypes() {
        return new ArrayList<>();
    }

    @Override
    public NexusRowFilter<NDataProvider.ClassificationProvider> createRowFilter(NexusTableModel tableModel, String name) {
        return null;
    }


    @Override
    public void addColumnDataListener(ColumnDataListener cdl) {

    }

    @Override
    public boolean removeColumnDataListener(ColumnDataListener cdl) {
        return false;
    }


    public class ClassificationRenderer extends AbstractCellEditor implements TableCellEditor , TableCellRenderer {

        private JComponent jp_editor = new JPanel();

        public Component getTableCellRendererComponent (JTable table,
                                                        Object value,
                                                        boolean isSelected,
                                                        boolean isFocus,
                                                        int row,
                                                        int column)
        {


            if(table instanceof NexusTable) {
                NexusTable nt = ((NexusTable) table);
                //System.out.println("mkay");
                //jp_editor = NexusTable.getDefaultEditorBackgroundPanel(nt,nt.getTableModel().getHighlightingAndSelectionStatus(row));
                NexusTable.NexusInteractiveEditorInfrastructure editorInfra = nt.createInteractiveEditorInfrastructure(row);
                jp_editor = editorInfra.panel;
                //jp_editor = NexusTable.getDefaultEditorBackgroundPanel(nt,nt.getTableModel().getHighlightingAndSelectionStatus(row));
            }
            else {
                jp_editor = new JPanel();
            }


            // As a safety check, it's always good to verify the type of
            // value.
            if(value == null) {
                jp_editor.removeAll();
            }

            if (value instanceof List)
            {
                List<NClassification.NClass> classes = new ArrayList<>();
                ((List)value).stream().filter( ci -> ci instanceof NClassification.NClass).forEach( ci -> classes.add((NClassification.NClass)ci) );

                this.jp_editor.removeAll();
                this.jp_editor.setBorder(BorderFactory.createEmptyBorder(4,4,4,4));
                this.jp_editor.setLayout( new GridLayout(classes.size(),1) );

                for(int zi=0;zi<classes.size();zi++) {
                    JColorLabel cla = new JColorLabel(classes.get(zi).getName(),classes.get(zi).getColor());
                    this.jp_editor.add(cla);
                }

//                MouseAdapter mli = new MouseAdapter() {
//                    @Override
//                    public void mouseEntered(MouseEvent e) {
//                        if( e.getComponent() instanceof JComponent) {
//                            ((JComponent)e.getComponent()).setBorder(new LineBorder(Color.blue,1));
//                        }
//                    }
//                    @Override
//                    public void mouseExited(MouseEvent e) {
//                        ((JComponent)e.getComponent()).setBorder(null);
//                    }
//                };
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
            return this.jp_editor;
        }

        private Object lastValue = null;
        @Override
        public Component getTableCellEditorComponent(JTable table, Object value, boolean isSelected, int row, int column) {

            System.out.println("Create cell editor: "+row+"/"+column);
            this.lastValue = value;
            Component ci = getTableCellRendererComponent(table,value,isSelected,true,row,column);


            //ci.addMouseListener(new DispatchingMouseAdapter(()->ci.getParent()));
            //JPanel p_editor = new JPanel();
            //p_editor.setLayout(new BorderLayout());

            if(ci instanceof JComponent) {
            //if(true) {
                JComponent p_editor = (JComponent) ci;
                JPopupMenu pop = new JPopupMenu();
                //String ridi = ntm.getRowIdForVisibleRow(row);
                for (CellSpecificAction csi : cellPopupActions) {
                    //csi.setRowId( () -> Collections.singletonList(ridi) );
                    //CellSpecificAction csia = csi.createArmedVersion(csi.getActionName(),()->Collections.singletonList(ridi));
                    pop.add(csi);
                }
                //pop.add(new ClusterAppModel.CreateClusterAction("Create Cluster",()-> Pair.of("cx",Color.blue),()->Collections.singletonList(ridi) ));
                //p_editor.setComponentPopupMenu(pop);

                ((JComponent) ci).setComponentPopupMenu(pop); // hmm.. this does somehow break the click selection stuff of the jtable.. :(
                //((JComponent) ci).addMouseListener(new DispatchingMouseAdapter(()->table));
                return ci;
            }
            // we should not end up here
            System.out.println("[ERROR] we should not end up here (ClassificationColumn.ClassificationRenderer)");
            return ci;
            //p_editor.add(ci, BorderLayout.CENTER);
        }

        @Override
        public Object getCellEditorValue() {
            return this.lastValue;
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
