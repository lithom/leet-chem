package tech.molecules.leet.table;

import com.actelion.research.chem.IDCodeParser;
import com.actelion.research.chem.SSSearcher;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.descriptor.DescriptorHandlerLongFFP512;
import com.actelion.research.gui.JEditableStructureView;
import com.actelion.research.gui.StructureListener;
import tech.molecules.leet.chem.BitSetUtils;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.table.gui.LeetChemistryCellRenderer;

import javax.swing.*;
import javax.swing.table.TableCellEditor;
import javax.swing.table.TableCellRenderer;
import java.awt.*;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;

public class StructureColumn implements NColumn<NStructureDataProvider, NStructureDataProvider.StructureWithID> {

    private NStructureDataProvider dp;
    private boolean loadFFPs = false;
    private boolean loadFFPsComplete = false;
    private Map<String,long[]> fps_FFP = new HashMap<>();



    public StructureColumn(boolean loadFFPs) {
        this.loadFFPs = loadFFPs;
    }

    @Override
    public void setDataProvider(NStructureDataProvider dataprovider) {
        this.dp = dataprovider;
    }

    @Override
    public NStructureDataProvider getDataProvider() {
        return this.dp;
    }

    public void startAsyncReinitialization(NexusTableModel ntm) {
        if(loadFFPs) {
            this.loadFFPsComplete = false;
            this.loadFFPsAync(dp, ntm);
        }
    }

    private void loadFFPsAync(NStructureDataProvider dataprovider, NexusTableModel ntm) {
        ExecutorService esa = Executors.newCachedThreadPool();
        List<String> all_rows = ntm.getAllRows();
        int zi=0;
        List<Future> fti = new ArrayList<>();
        while(zi<all_rows.size()) {
            String rowkey = all_rows.get(zi);
            String midc   = dataprovider.getStructureData(all_rows.get(zi)).structure[0];
            fti.add( esa.submit(new Runnable() {
                @Override
                public void run() {
                    StereoMolecule mi = ChemUtils.parseIDCode(midc);
                    long ffpi[] = DescriptorHandlerLongFFP512.getDefaultInstance().getThreadSafeCopy().createDescriptor(mi);
                    synchronized(fps_FFP) {
                        fps_FFP.put(rowkey,ffpi);
                    }
                }
            }));
            zi++;
        }
        // wait for all thrads to compute, then set loadComplete.
        esa.submit(new Runnable(){
            @Override
            public void run() {
                for(Future fi : fti) {
                    try {
                        fi.get();
                    } catch (InterruptedException e) {
                        throw new RuntimeException(e);
                    } catch (ExecutionException e) {
                        throw new RuntimeException(e);
                    }
                }
                loadFFPsComplete = true;
            }
        });
    }

    @Override
    public String getName() {
        return "Structure";
    }

    @Override
    public NStructureDataProvider.StructureWithID getData(String rowid) {
        return dp.getStructureData(rowid);
    }

    @Override
    public Map<String, NumericalDatasource<NStructureDataProvider>> getNumericalDataSources() {
        return new HashMap<>();
    }

    //@Override
    public TableCellRenderer getCellRenderer() {
        return new StructureCellRenderer();
    }

    @Override
    public TableCellEditor getCellEditor() {
        return new StructureCellRenderer();
        //return new NexusTable.DefaultEditorFromRenderer(this.getCellRenderer());
    }

    private List<ColumnDataListener> listeners = new ArrayList<>();

    @Override
    public void addColumnDataListener(ColumnDataListener cdl) {
        listeners.add(cdl);
    }

    @Override
    public boolean removeColumnDataListener(ColumnDataListener cdl) {
        return this.listeners.remove(cdl);
    }


    public static String ROW_FILTER_SS = "rowfilter_ss";

    @Override
    public List<String> getRowFilterTypes() {
        List<String> rfs = new ArrayList<>();
        rfs.add(ROW_FILTER_SS);
        return rfs;
    }

    private List<NStructureDataProvider> providers = new ArrayList<>();

    @Override
    public NexusRowFilter<NStructureDataProvider> createRowFilter(NexusTableModel tableModel, String name) {
        if(name.equals(ROW_FILTER_SS)) {
            return new SubstructureRowFilter();
        }
        return null;
    }

    public static class StructureCellRenderer extends LeetChemistryCellRenderer { //extends ChemistryCellRenderer {


        @Override
        public Component getTableCellEditorComponent(JTable table, Object value, boolean isSelected, int row, int col) {
            if(value instanceof String) {
                return super.getTableCellEditorComponent(table, (String) value, isSelected, row, col);
            }
            if(value instanceof NStructureDataProvider.StructureWithID) {
                String idc[] = ((NStructureDataProvider.StructureWithID) value).structure;
                //return new JLabel(idc[0]);
                //return super.getTableCellRendererComponent(table, idc[0]+" "+idc[1], isSelected, hasFocus, row, col);
                return super.getTableCellEditorComponent(table, idc[0] + " " + idc[1], isSelected, row, col);
            }
            //return new JLabel("<NoData>");
            return super.getTableCellEditorComponent(table, value, isSelected, row, col);
        }

//        @Override
//        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int col) {
//            if(value instanceof String) {
//                return super.getTableCellEditorComponent(table, (String) value, isSelected, row, col);
//            }
//            if(value instanceof NStructureDataProvider.StructureWithID) {
//                String idc[] = ((NStructureDataProvider.StructureWithID) value).structure;
//                //return new JLabel(idc[0]);
//                //return super.getTableCellRendererComponent(table, idc[0]+" "+idc[1], isSelected, hasFocus, row, col);
//                return super.getTableCellEditorComponent(table, idc[0] + " " + idc[1], isSelected, row, col);
//            }
//            return new JLabel("<NoData>");
//        }
    }

    public class SubstructureRowFilter implements NexusRowFilter<NStructureDataProvider> {

        private NexusTableModel ntm;
        private StereoMolecule filterStructure = null;
        private long[]         filterStructureFFP = new long[16];
        @Override
        public String getFilterName() {
            return ROW_FILTER_SS;
        }

        @Override
        public BitSet filterNexusRows(NStructureDataProvider data, List<String> ids, BitSet filtered) {
            if(!this.isReady() || this.filterStructure==null ) {
                return (BitSet) filtered.clone();
            }
            IDCodeParser icp = new IDCodeParser();
            SSSearcher sss = new SSSearcher();
            BitSet f2 = (BitSet) filtered.clone();
            sss.setFragment(this.filterStructure);
            for(int zi=0;zi<ids.size();zi++) {
                if(!f2.get(zi)) {continue;}
                boolean fphit = BitSetUtils.test_subset(this.filterStructureFFP,fps_FFP.get(ids.get(zi)));
                if(fphit) {
                    StereoMolecule mmi = new StereoMolecule();
                    icp.parse(mmi,data.getStructureData(ids.get(zi)).structure[0]);
                    sss.setMolecule(mmi);
                    f2.set(zi,sss.isFragmentInMolecule());
                }
                else {
                    f2.set(zi,false);
                }
            }
             return f2;
        }

        @Override
        public double getApproximateFilterSpeed() {
            return 0.2;
        }

        @Override
        public void setupFilter(NexusTableModel model, NStructureDataProvider dp) {
            this.ntm = model;
            if(!loadFFPs) {
                loadFFPs = true;
                startAsyncReinitialization(model);
            }
        }

        @Override
        public JPanel getFilterGUI() {
            return new JSSFilterpanel();
        }

        @Override
        public boolean isReady() {
            return loadFFPsComplete;
        }

        public void setFilterStructure(StereoMolecule mi) {
            this.filterStructure = mi;
            if(!this.filterStructure.isFragment()) {
                this.filterStructure.setFragment(true);
            }
            this.filterStructureFFP = DescriptorHandlerLongFFP512.getDefaultInstance().getThreadSafeCopy().createDescriptor(this.filterStructure);
            ntm.updateFiltering();
        }

        class JSSFilterpanel extends JPanel {
            public JSSFilterpanel() {
                JEditableStructureView ssview = new JEditableStructureView();
                ssview.setAllowQueryFeatures(true);
                this.setLayout(new BorderLayout());
                this.add(ssview,BorderLayout.CENTER);
                ssview.addStructureListener(new StructureListener() {
                    @Override
                    public void structureChanged(StereoMolecule stereoMolecule) {
                        setFilterStructure(stereoMolecule);
                    }
                });
                this.setPreferredSize(new Dimension(200,200));
                this.repaint();
            }
        }

    }

}
