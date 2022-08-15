package tech.molecules.leet.table.chem;

import com.actelion.research.chem.IDCodeParser;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.properties.ChemPropertyCounts;
import tech.molecules.leet.table.NColumn;
import tech.molecules.leet.table.NStructureDataProvider;
import tech.molecules.leet.table.NexusTableModel;
import tech.molecules.leet.table.gui.JExtendedSlimRangeSlider;
import tech.molecules.leet.table.gui.JSlimRangeSlider;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.util.*;
import java.util.List;

public class NexusChemPropertiesFilter implements NColumn.NexusRowFilter<NStructureDataProvider> {

    private NexusTableModel model;

    private boolean removeConnectors = true;
    private NColumn<NStructureDataProvider.StructureWithID,NStructureDataProvider> col;

    public NexusChemPropertiesFilter(NColumn<NStructureDataProvider.StructureWithID,NStructureDataProvider> column) {
        this.col = column;
        this.setConsideredCounts(Arrays.asList(ChemPropertyCounts.COUNTS_ALL));
    }

    public void setRemoveConnectors(boolean remove) {
        this.removeConnectors = remove;
    }

    private List<ChemPropertyCounts.ChemPropertyCount> consideredCounts = new ArrayList<>();

    // total range of values that we find in the dataset
    private Map<ChemPropertyCounts.ChemPropertyCount, int[]> ranges = new HashMap<>();

    // restriction range for filtering
    private Map<ChemPropertyCounts.ChemPropertyCount, JExtendedSlimRangeSlider> bounds = new HashMap<>();

    public void setConsideredCounts(List<ChemPropertyCounts.ChemPropertyCount> counts) {
        this.consideredCounts = counts;
    }

    @Override
    public String getFilterName() {
        return "Chemical Properties Filter";
    }

    @Override
    public BitSet filterNexusRows(NStructureDataProvider data, List<String> ids, BitSet filtered) {

        BitSet filtered_2 = (BitSet) filtered.clone();

        IDCodeParser icp = new IDCodeParser();
        for(int zi=0;zi<ids.size();zi++) {
            if(!filtered.get(zi)) {
                //filtered_2.set(zi);
                continue;
            }

            StereoMolecule mi = new StereoMolecule();
            NStructureDataProvider.StructureWithID s = data.getStructureData(ids.get(zi));
            icp.parse(mi,s.structure[0],s.structure[1]);
            mi.ensureHelperArrays(Molecule.cHelperCIP);

            for( ChemPropertyCounts.ChemPropertyCount ci : this.consideredCounts) {
                int mci = ci.evaluator.apply(mi);
                JSlimRangeSlider sli = this.bounds.get(ci).getRangeSlider();
                if(sli==null) {
                    System.out.println("[ERROR] range slider in filter missing..");
                }
                double ri[] = sli.getRange();
                if(mci>= ri[0] && mci<=ri[1]) {

                }
                else {
                    filtered.set(zi,false);
                    filtered_2.set(zi,false);
                }
            }

        }
        return filtered_2;
    }

    @Override
    public double getApproximateFilterSpeed() {
        return 0.4;
    }

    @Override
    public void setupFilter(NexusTableModel model, NStructureDataProvider dp) {
        this.model = model;
        // init ranges
        initRanges(model);
        initGUI();
    }

    private void initRanges(NexusTableModel model) {
        NStructureDataProvider data = model.getDatasetForColumn(this.col);
        IDCodeParser icp = new IDCodeParser();
        for(ChemPropertyCounts.ChemPropertyCount ci : this.consideredCounts) {
            int min=1000000; int max=-1000000;
            for (String ri : model.getAllRows()) {
                StereoMolecule mi = new StereoMolecule();
                icp.parse(mi,data.getStructureData(ri).structure[0]);
                mi.ensureHelperArrays(Molecule.cHelperCIP);
                int cci = ci.evaluator.apply(mi);
                min = Math.min(min,cci);
                max = Math.max(max,cci);
            }
            this.ranges.put(ci,new int[]{min,max});
        }
    }

    @Override
    public boolean isReady() {
        return true;
    }

    private JPanel jFilterPanel;

    private void initGUI() {
        jFilterPanel = new JPanel();
        this.jFilterPanel.setLayout(new BorderLayout());

        int totalNumberOfProperties = this.consideredCounts.size();

        JPanel jp = new JPanel();
        int rows = 4; int cols = 4;
        jp.setLayout(new GridLayout(rows,cols));

        // init count filters:
        for(int zi=0;zi<this.consideredCounts.size();zi++) {
            JExtendedSlimRangeSlider jsi = new JExtendedSlimRangeSlider(this.consideredCounts.get(zi).name,new double[]{ ranges.get(this.consideredCounts.get(zi))[0] , ranges.get(this.consideredCounts.get(zi))[1] });
            bounds.put(this.consideredCounts.get(zi),jsi);
            jp.add(jsi);

            jsi.getRangeSlider().addChangeListener(new ChangeListener() {
                @Override
                public void stateChanged(ChangeEvent e) {
                    model.updateFiltering();
                }
            });
        }

        this.jFilterPanel.add(jp,BorderLayout.CENTER);
    }

    @Override
    public JPanel getFilterGUI() {
        return jFilterPanel;
    }
}
