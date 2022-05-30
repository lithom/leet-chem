package tech.molecules.leet.table;

import com.actelion.research.chem.IDCodeParser;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.PropertyCalculator;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.gui.DefaultCompoundCollectionModel;

import javax.swing.*;
import javax.swing.border.LineBorder;
import javax.swing.event.CellEditorListener;
import javax.swing.table.TableCellEditor;
import javax.swing.table.TableCellRenderer;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.EventObject;
import java.util.List;

public class StructureCalculatedPropertiesColumn implements NColumn<NStructureDataProvider, StructureCalculatedPropertiesColumn.CalculatedProperties> {

    public static class CalculatedProperties {
        public final int hac;
        public final int rbs;
        public final int acc;
        public final int don;

        public CalculatedProperties(int hac, int rbs, int acc, int don) {
            this.hac = hac;
            this.rbs = rbs;
            this.acc = acc;
            this.don = don;
        }
    }


    public StructureCalculatedPropertiesColumn() {

    }

    @Override
    public String getName() {
        return "Properties";
    }

    @Override
    public CalculatedProperties getData(NStructureDataProvider data, String rowid) {
        NStructureDataProvider.StructureWithID sid = data.getStructureData(rowid);
        StereoMolecule m = new StereoMolecule();
        IDCodeParser icp = new IDCodeParser();
        icp.parse(m,sid.structure[0],sid.structure[1]);
        m.ensureHelperArrays(Molecule.cHelperCIP);
        PropertyCalculator pc = new PropertyCalculator(m);
        return new CalculatedProperties(m.getAtoms(),m.getRotatableBondCount(),pc.getAcceptorCount(),pc.getDonorCount());
    }

    @Override
    public List<String> getNumericalDataSources() {
        return new ArrayList<>();
    }

    @Override
    public double evaluateNumericalDataSource(NStructureDataProvider dp, String datasource, String rowid) {
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



    public static class CalcPropertiesRenderer extends JPanel implements TableCellEditor , TableCellRenderer {


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

            if (value instanceof CalculatedProperties)
            {
                //String s = (String) value;
                CalculatedProperties a = (CalculatedProperties) value;

                this.removeAll();
                this.setLayout( new GridLayout(4,1) );


                JColorLabel pa = new JColorLabel( String.format("hac= %d",a.hac),a.hac,0,40);
                JColorLabel pb = new JColorLabel( String.format("rbs= %d",a.rbs),a.rbs,0,16);
                JLabel pc = new JColorLabel( String.format("acc %d",a.acc),a.acc,0,10);
                JLabel pd = new JColorLabel( String.format("don %d",a.don),a.don,0,10);

                this.add(pa);this.add(pb);
                this.add(pc);this.add(pd);

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
            return getTableCellRendererComponent(table,value,isSelected,true,row,column);
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
                this.setBackground(c);
            }
        }

    }

}
