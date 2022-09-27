package tech.molecules.leet.table;

import com.actelion.research.chem.IDCodeParser;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.descriptor.DescriptorHandler;

import javax.swing.table.TableCellEditor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PairwiseDistanceColumn implements NSimilarityColumn<NDataProvider.StructureDataProvider, StereoMolecule> {

    private NDataProvider.StructureDataProvider dp;
    private Map<String,Integer> rowPos;
    private double[][] dist;

    private boolean isSimilarity;
    private boolean isNormalized;
    private boolean isSymmetric;

    public PairwiseDistanceColumn(Map<String,Integer> row_pos, double[][] distances, boolean isSimilarity, boolean isNormalized, boolean isSymmetric) {
        this.rowPos = row_pos;
        this.dist = distances;

        this.isSimilarity = isSimilarity;
        this.isNormalized = isNormalized;
        this.isSymmetric  = isSymmetric;
    }

    @Override
    public String getName() {
        return "dist..";
    }

    @Override
    public void setDataProvider(NDataProvider.StructureDataProvider dataprovider) {
        this.dp = dataprovider;
    }

    @Override
    public NDataProvider.StructureDataProvider getDataProvider() {
        return this.dp;
    }

    @Override
    public void startAsyncReinitialization(NexusTableModel model) {

    }

    @Override
    public StereoMolecule getData(String rowid) {
        return null;
    }

    @Override
    public TableCellEditor getCellEditor() {
        return null;
    }

    @Override
    public Map<String, NumericalDatasource<NDataProvider.StructureDataProvider>> getNumericalDataSources() {
        return new HashMap<>();
    }

    @Override
    public void addCellPopupAction(CellSpecificAction ca) {
        NSimilarityColumn.super.addCellPopupAction(ca);
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

    @Override
    public boolean isSimilarity() {
        return isSimilarity;
    }

    @Override
    public boolean isNormalized() {
        return isNormalized;
    }

    @Override
    public boolean isSymmetric() {
        return isSymmetric;
    }

    @Override
    public double evaluateValue(NDataProvider.StructureDataProvider data, String rowid_a, String rowid_b) {
        Integer pa = this.rowPos.get(rowid_a);
        Integer pb = this.rowPos.get(rowid_b);
        if(pa==null || pb==null) {return Double.NaN;}
        return this.dist[pa][pb];
    }

    public static PairwiseDistanceColumn createFromDescriptor(NDataProvider.StructureDataProvider data, DescriptorHandler dh, List<String> rows) {
        Map<String,Integer> pos = new HashMap<>();
        List<Object> dhs = new ArrayList<>();

        IDCodeParser icp = new IDCodeParser();
        for(int zi=0;zi<rows.size();zi++) {
            pos.put(rows.get(zi),zi);
            StereoMolecule mi = new StereoMolecule();
            //icp.parse(mi,rows.get(zi));
            icp.parse(mi,data.getStructureData(rows.get(zi)).structure[0]);
            mi.ensureHelperArrays(Molecule.cHelperCIP);
            dhs.add( dh.createDescriptor(mi) );
        }

        double[][] distdata = new double[rows.size()][rows.size()];
        for(int zi=0;zi<rows.size();zi++) { distdata[zi][zi]=1.0; }
        for(int zi=0;zi<rows.size()-1;zi++) {
            for(int zj=zi+1;zj<rows.size();zj++) {
                double di = dh.getSimilarity(dhs.get(zi),dhs.get(zj));
                distdata[zi][zj] = di;
                distdata[zj][zi] = di;
            }
        }
        return new PairwiseDistanceColumn(pos,distdata,true,true,true);
    }

}
