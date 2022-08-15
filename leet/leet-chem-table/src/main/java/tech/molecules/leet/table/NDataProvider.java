package tech.molecules.leet.table;

import com.actelion.research.chem.reaction.Classification;

import java.io.Serializable;
import java.util.List;

public interface NDataProvider {

    public static class StructureWithID implements Serializable {
        public final String structure[];
        public final String molid;
        public final String batchid;

        public StructureWithID(String molid,String batchid,String[] struc) {
            this.structure = struc;
            this.molid = molid;
            this.batchid = batchid;
        }
    }

    public static interface StructureDataProvider {
        public StructureWithID getStructureData(String rowid);
    }

    public static class StructureDataProviderFromColumn<U> implements NStructureDataProvider {
        private U dataProvider;
        private NColumn<U,StructureWithID> col;
        public StructureDataProviderFromColumn(U dataprovider, NColumn<U,StructureWithID> col) {
            this.dataProvider = dataprovider;
            this.col = col;
        }
        @Override
        public StructureWithID getStructureData(String rowid) {
            return this.col.getData(this.dataProvider,rowid);
        }
    }

    public static interface ClassificationDataProvider extends NDataProvider {
        //public List<Classification.NClass>
        public List<NClassification.NClass> getClassificationData(String rowid);
    }

    public static interface StringDataProvider {
        public String getStringData(String rowid);
    }

    public static interface NStringDataProvider extends NDataProvider , StringDataProvider
    {

    }

    public static interface MultiNumericDataProvider {
        public double[] getMultiNumericData(String rowid);
    }

    public static interface NMultiNumericDataProvider extends NDataProvider , MultiNumericDataProvider {

    }

}
