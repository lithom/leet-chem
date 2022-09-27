package tech.molecules.leet.table;

import com.actelion.research.chem.IDCodeParser;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.PropertyCalculator;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.gui.DefaultCompoundCollectionModel;
import tech.molecules.leet.chem.mutator.properties.ChemPropertyCounts;
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
import java.util.function.Supplier;

public class StructureCalculatedPropertiesColumn implements NColumn<NStructureDataProvider, StructureCalculatedPropertiesColumn.CalculatedProperties> {

    private NStructureDataProvider dp;

    public static class CalculatedProperties {
        /**
         * indeces are positions in ChemPropertyCounts.COUNTS_ALL
         */
        public Map<Integer,Integer> counts = new HashMap<>();

        public CalculatedProperties(Map<Integer,Integer> counts) {
            this.counts = new HashMap<>(counts);
        }
    }

    @Override
    public void setDataProvider(NStructureDataProvider dataprovider) {
        this.dp = dataprovider;
    }

    @Override
    public NStructureDataProvider getDataProvider() {
        return this.dp;
    }

    @Override
    public void startAsyncReinitialization(NexusTableModel model) {

    }

    public StructureCalculatedPropertiesColumn() {

    }

    @Override
    public String getName() {
        return "Properties";
    }

    @Override
    public CalculatedProperties getData(String rowid) {
        NStructureDataProvider.StructureWithID sid = dp.getStructureData(rowid);
        StereoMolecule m = new StereoMolecule();
        IDCodeParser icp = new IDCodeParser();
        icp.parse(m,sid.structure[0],sid.structure[1]);
        m.ensureHelperArrays(Molecule.cHelperCIP);

        Map<Integer,Integer> counts = new HashMap<>();
        for(int zi=0;zi< ChemPropertyCounts.COUNTS_ALL.length;zi++) {
            counts.put( zi, ChemPropertyCounts.COUNTS_ALL[zi].evaluator.apply(m) );
        }
        return new CalculatedProperties(counts);
    }

    public StructureCalculatedPropertiesColumn getThisColumn() {
        return this;
    }

    @Override
    public Map<String, NumericalDatasource<NStructureDataProvider>> getNumericalDataSources() {
        Map<String,NumericalDatasource<NStructureDataProvider>> nds = new HashMap<>();
        for(int zi=0;zi< ChemPropertyCounts.COUNTS_ALL.length;zi++) {
            nds.put( ChemPropertyCounts.COUNTS_ALL[zi].name, new ChemPropertyCountDatasource(ChemPropertyCounts.COUNTS_ALL[zi]) );
        }
        return nds;
    }

    public class ChemPropertyCountDatasource implements NumericalDatasource<NStructureDataProvider> {
        private ChemPropertyCounts.ChemPropertyCount ci;

        public ChemPropertyCountDatasource(ChemPropertyCounts.ChemPropertyCount ci) {
            this.ci = ci;
        }

        @Override
        public String getName() {
            return this.ci.name;
        }

        @Override
        public NColumn<NStructureDataProvider, ?> getColumn() {
            return getThisColumn();
        }

        @Override
        public boolean hasValue(String row) {
            return true;
        }

        @Override
        public double getValue(String row) {
            StereoMolecule m = new StereoMolecule();
            IDCodeParser icp = new IDCodeParser();
            icp.parse(m,dp.getStructureData(row).structure[0],dp.getStructureData(row).structure[1]);
            m.ensureHelperArrays(Molecule.cHelperCIP);
            return this.ci.evaluator.apply(m);
        }
    }


    public static double evaluateNumericalDataSource(NStructureDataProvider dp, String datasource, String rowid) {
        StereoMolecule mi = new StereoMolecule();
        IDCodeParser icp = new IDCodeParser();
        icp.parse(mi,dp.getStructureData(rowid).structure[0],dp.getStructureData(rowid).structure[1]);
        mi.ensureHelperArrays(Molecule.cHelperCIP);
        switch(datasource) {
            case "hac": return mi.getAtoms();
            case "rb": return mi.getRotatableBondCount();
        }
        return Double.NaN;
    }

    //@Override
    public TableCellRenderer getCellRenderer() {
        return new CalcPropertiesRenderer();
    }

    public TableCellEditor getCellEditor() {
        return new CalcPropertiesRenderer();
    }

    @Override
    public List<String> getRowFilterTypes() {
        return null;
    }

    @Override
    public NexusRowFilter<NStructureDataProvider> createRowFilter(NexusTableModel tableModel, String name) {
        return null;
    }

    @Override
    public void addColumnDataListener(ColumnDataListener cdl) {

    }

    @Override
    public boolean removeColumnDataListener(ColumnDataListener cdl) {
        return false;
    }



    public static class CalcPropertiesRenderer extends AbstractCellEditor implements TableCellEditor , TableCellRenderer {


        public Component getTableCellRendererComponent (JTable table,
                                                        Object value,
                                                        boolean isSelected,
                                                        boolean isFocus,
                                                        int row,
                                                        int column)
        {
            //JPanel pi = new JPanel();
            NexusTable nt = (NexusTable) table;
            NexusTable.JCellBackgroundPanel pi = NexusTable.getDefaultEditorBackgroundPanel(nt,nt.getTableModel().getHighlightingAndSelectionStatus(row));

            // As a safety check, it's always good to verify the type of
            // value.

            if(value == null) {
                pi.removeAll();
            }

            //if(value != null) {
                //JLabel jl = new JLabel("TEST!!");
                //pi.setLayout(new FlowLayout()); pi.add(jl);
            //    return pi;
            //}

            if (value instanceof CalculatedProperties) {
                //String s = (String) value;
                CalculatedProperties a = (CalculatedProperties) value;

                pi.removeAll();
                pi.setLayout(new GridLayout(4, 4));


                for (int zi = 0; zi < ChemPropertyCounts.COUNTS_ALL.length; zi++) {
                    JColorLabel pa = new JColorLabel(ChemPropertyCounts.COUNTS_ALL[zi].shortName+"="+String.format("%d", a.counts.get(zi)), a.counts.get(zi), 0, 40);

                    pi.add(pa);

                    //Supplier<Component> componentSupplier = (pi!=null) ? () -> pi : ()->getParent();
                    Supplier<Component> componentSupplier = () -> pi.getParent();

                    DispatchingMouseAdapter mli = new DispatchingMouseAdapter(componentSupplier) {
                        @Override
                        public void mouseEntered(MouseEvent e) {
//                            if (e.getComponent() instanceof JComponent) {
//                                ((JComponent) e.getComponent()).setBorder(new LineBorder(Color.blue, 1));
//                            }
                            pa.setBorder(new LineBorder(Color.blue, 1));
                            super.mouseEntered(e);
                        }

                        @Override
                        public void mouseExited(MouseEvent e) {
                            //((JComponent) e.getComponent()).setBorder(null);
                            pa.setBorder(null);
                            super.mouseExited(e);
                        }
                    };
                    pa.addMouseListener(mli);
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
            }
            // Return a reference to the ColorRenderer (JLabel subclass)
            // component. Behind the scenes, that component will be used to
            // paint the pixels.
            return pi;
        }

        private Object lastValue = null;
        @Override
        public Component getTableCellEditorComponent(JTable table, Object value, boolean isSelected, int row, int column) {
            this.lastValue = value;
            return getTableCellRendererComponent(table,value,isSelected,true,row,column);
        }

        @Override
        public Object getCellEditorValue() {
            return this.lastValue;
        }

        public static class JColorLabel extends JLabel {
            public JColorLabel(String text, double val, double va, double vb) {
                super(text);
                double f = (val-va) / (vb-va);

                float value = (float) Math.max( 0 , Math.min(1 , f) ) ; //this is your value between 0 and 1
                float minHue = 120f/255; //corresponds to green
                float maxHue = 0; //corresponds to red
                float hue = value*maxHue + (1-value)*minHue;
                Color ca = new Color( Color.HSBtoRGB(hue, 0.6f, 0.85f ) );
                Color c = new Color( ca.getRed() , ca.getGreen() , ca.getBlue() , 40 );
                this.setOpaque(true);
                //this.setOpaque(false);
                this.setBackground(c);
            }
        }

    }

}
