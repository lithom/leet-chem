package tech.molecules.leet.datatable.swing.chem;

import com.actelion.research.chem.Canonizer;
import com.actelion.research.chem.StereoMolecule;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.StructureWithID;
import tech.molecules.leet.chem.sar.SimpleSARDecomposition;
import tech.molecules.leet.chem.sar.SimpleSARDecompositionModel;
import tech.molecules.leet.chem.sar.SimpleSARSeries;
import tech.molecules.leet.datatable.DataProvider;
import tech.molecules.leet.datatable.DataTable;
import tech.molecules.leet.datatable.chem.StructureWithIDColumn;
import tech.molecules.leet.datatable.column.AbstractDataTableColumn;
import tech.molecules.leet.datatable.dataprovider.DataProviderListenerHelper;

import java.util.*;
import java.util.stream.Collectors;

public class SimpleSARTableModel {

    /**
     * Returns results for one series of the SimpleSARDecomposionModel
     *
     */
    public static class SARDataProvider implements DataProvider<SimpleSARDecomposition.SimpleSARResult> {
        private SimpleSARDecompositionModel decompModel;
        private SimpleSARSeries series;
        private DataProviderListenerHelper listenerHelper = new DataProviderListenerHelper();
        private Map<String, SimpleSARDecomposition.SimpleSARResult> knownKeysWithData = new HashMap<>();

        public SARDataProvider(SimpleSARDecompositionModel model, SimpleSARSeries series) {
            this.decompModel = model;
            this.series = series;

                this.decompModel.addListener(new SimpleSARDecompositionModel.DecompositionModelListener() {
                    @Override
                    public void newDecompositions() {
                        List<SimpleSARDecomposition.SimpleSARResult> results_all = decompModel.getSeriesDecomposition(series);
                        List<String> results_new = results_all.stream().map(xi -> xi.getStructure()[0]).collect(Collectors.toList());
                        results_new.removeAll(knownKeysWithData.keySet());
                        synchronized (knownKeysWithData) {
                            //knownKeys.addAll(results_new);
                            results_all.stream().forEach( xi -> knownKeysWithData.put(xi.getStructure()[0],xi) );
                        }
                        listenerHelper.fireDataChanged(results_new);
                    }
                });

        }

        @Override
        public List<String> getAllEntries() {
            List<String> keys = new ArrayList<>();
            synchronized(knownKeysWithData) {
                keys = new ArrayList<>(this.knownKeysWithData.keySet());
            }
            return keys;
        }

        @Override
        public SimpleSARDecomposition.SimpleSARResult getData(String key) {
            //return new StructureWithID(key,"",ChemUtils.parseIDCode(key));
            SimpleSARDecomposition.SimpleSARResult ri = null;
            synchronized(knownKeysWithData) {
                ri = knownKeysWithData.get(key);
            }
            return ri;
        }

        @Override
        public void addDataProviderListener(DataProviderListener li) {
            listenerHelper.addDataProviderListener(li);
        }

        @Override
        public boolean removeDataProviderListener(DataProviderListener li) {
            return listenerHelper.removeDataProviderListener(li);
        }
    }


    public static class SARColumn extends AbstractDataTableColumn<SimpleSARDecomposition.SimpleSARResult, StructureWithID> {

        /**
         * indicates which fragment to show.
         * null means show the full structure
         */
        private String matchedFrag;

        public SARColumn(String matchedFrag) {
            super(StructureWithID.class);
            this.matchedFrag = matchedFrag;
        }

        @Override
        public StructureWithID processData(SimpleSARDecomposition.SimpleSARResult data) {
            if(this.matchedFrag==null) {
                return new StructureWithID( data.getStructure()[0] , "" , data.getStructure());
            }
            SimpleSARDecomposition.MatchedFragment mfi = data.get(this.matchedFrag);
            if(mfi==null) {return new StructureWithID("","",new StereoMolecule());}
            Canonizer ci = new Canonizer(mfi.matchedFrag,Canonizer.ENCODE_ATOM_CUSTOM_LABELS);
            String idc = ci.getIDCode();
            //return new StructureWithID( mfi.matchedFrag );
            return new StructureWithID( idc, "", new String[]{idc,ci.getEncodedCoordinates()} );
        }
    }

    private DataTable table;

    private SimpleSARDecompositionModel model;
    private SimpleSARSeries series;

    public void reinitTable(SimpleSARDecompositionModel model, SimpleSARSeries series) {
        this.model = model;
        this.series = series;


        //DefaultStructureProvider sid_col_dp = new DefaultStructureProvider();
        //List<SimpleSARDecomposition.SimpleSARResult> results = this.model.getSeriesDecomposition(series);
        //List<StructureWithID> r_as_sids = results.stream().map( xi -> new StructureWithID(xi.getStructure(),"", ChemUtils.parseIDCode(xi.getStructure()))).collect(Collectors.toList());
        //sid_col_dp.loadStructures( r_as_sids );
        SARDataProvider sid_col_dp = new SARDataProvider(this.model,this.series);

        table = new DataTable();
        //StructureWithIDColumn sid_col = new StructureWithIDColumn();

        List<SARColumn> columns = new ArrayList<>();
        SARColumn col_id = new SARColumn(null);
        columns.add(col_id);
        for(String si : this.series.getLabels()) {
            columns.add(new SARColumn(si));
        }

        columns.forEach(ci -> table.addDataColumn(ci));
        columns.forEach(ci -> ci.setDataProvider(sid_col_dp));


        table.setAllKeys( sid_col_dp.getAllEntries() );
        sid_col_dp.addDataProviderListener(new DataProvider.DataProviderListener() {
            @Override
            public void dataChanged(List<String> keysChanged) {
                table.setAllKeys(sid_col_dp.getAllEntries());
            }
        });
    }

    public DataTable getDataTable() {return this.table;}

}
