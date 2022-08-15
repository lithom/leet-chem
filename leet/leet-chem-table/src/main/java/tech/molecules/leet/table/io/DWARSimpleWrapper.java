package tech.molecules.leet.table.io;


import com.actelion.research.chem.IDCodeParser;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.SmilesParser;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.chem.io.DWARFileParser;
import com.actelion.research.util.ConstantsDWAR;
import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.table.*;

import java.io.BufferedReader;
import java.sql.Array;
import java.util.*;

/**
 * Reads a dwar file into memory
 *
 */
public class DWARSimpleWrapper {

    //public static enum DWARColType { STRUCTURE, DESCRIPTOR, TEXT, NUMERIC, NUMERIC_ARRAY };
    //public static class ImportColumn {
    //    DWARColType type;
    //    String name;
    //    String parentName;
    //}


    public interface DWARColumnImporter<U extends NDataProvider,T extends NColumn> {
        public void prepare(DWARFileParser dw);
        public void readLine(String key, DWARFileParser dw);
        public void postprocess(DWARFileParser dw);
        public boolean canParse(String data);
        public U getDataProvider();
        public T getColumn();
    }

    //public abstract class DWARAbstractColumnImporter<U extends NDataProvider,T extends NColumn> implements DWARColumnImporter {
    //    public DWARAbstractColumnImporter() {
    //
    //    }
    //}

    public static class StructureColumnImporter implements DWARColumnImporter<NStructureDataProvider, StructureColumn> {
        private String colname;
        private boolean specialfield;
        private int    cidx = -1;
        private IDCodeParser icp = new IDCodeParser(true);
        private DefaultStructureDataProvider dsdp = null;
        private List<String[]> structuredata = new ArrayList<>();
        public StructureColumnImporter(String colname, boolean specialfield) {
            this.colname = colname;
            this.specialfield = specialfield;
        }

        @Override
        public boolean canParse(String data) {
            StereoMolecule mi = parseMoleculeData(data);
            return mi!=null;
            //mi.ensureHelperArrays(Molecule.cHelperCIP);
            //return mi.getAtoms()>0;
        }

        @Override
        public void prepare(DWARFileParser dw) {
            if(!this.specialfield) {
                this.cidx = dw.getFieldIndex(this.colname);
            }
            else {
                this.cidx = dw.getSpecialFieldIndex(this.colname);
            }
            this.structuredata = new ArrayList<>();
        }

        @Override
        public void readLine(String key, DWARFileParser dw) {
            String di = (this.specialfield)?dw.getSpecialFieldData(this.cidx):dw.getFieldData(this.cidx);
            di = di.trim();
            StereoMolecule mi = parseMoleculeData(di);
            if(mi!=null) {
                this.structuredata.add(new String[]{key, mi.getIDCode()});
            }
            else {
                this.structuredata.add(new String[]{key, ""});
            }
        }

        private StereoMolecule parseMoleculeData(String di) {
            StereoMolecule mi = null;//new StereoMolecule();
            boolean parsed = false;
            if(!di.isEmpty()) {
                StereoMolecule mi2 = new StereoMolecule();
                try{
                    icp.parse(mi2,di);
                    mi2.ensureHelperArrays(Molecule.cHelperCIP);
                    parsed = mi2.getAtoms()>0;
                    mi = mi2;
                }
                catch(Exception ex){
                    mi = null;//new StereoMolecule();
                }
                if(!parsed) {
                    try {
                        mi = (new SmilesParser()).parseMolecule(di);
                        mi.ensureHelperArrays(Molecule.cHelperCIP);
                        parsed = mi.getAtoms()>0; // for smiles we check that we did not just "parse" an empty molecule..
                        if(parsed) {
                            System.out.println("mkay");
                        }
                    } catch (Exception ex) {
                        mi = null;//new StereoMolecule();
                    }
                }
            }
            if(parsed) {
                return mi;
            }
            return null;
        }

        @Override
        public void postprocess(DWARFileParser dw) {
            this.dsdp = new DefaultStructureDataProvider(this.structuredata);
        }

        @Override
        public NStructureDataProvider getDataProvider() {
            return this.dsdp;
        }

        @Override
        public StructureColumn getColumn() {
            return new StructureColumn(true);
        }
    }

    public static class MultiNumericColumnImporter implements DWARColumnImporter<NDataProvider.NMultiNumericDataProvider, MultiNumericDataColumn> {
        private String colname;
        private boolean specialfield;
        private int    cidx = -1;
        private DefaultMultiNumericDataProvider mndp = null;
        Map<String,double[]> numdata = new HashMap<>();
        public MultiNumericColumnImporter(String colname, boolean specialf) {
            this.colname = colname;
            this.specialfield = specialf;
        }

        @Override
        public boolean canParse(String data) {
            double[] dval = parseNumericData(data.trim());
            if(dval.length==0) {return false;}
            return Arrays.stream(dval).anyMatch( di -> !Double.isNaN(di) );
        }

        @Override
        public void prepare(DWARFileParser dw) {
            if(!this.specialfield) {
                this.cidx = dw.getFieldIndex(this.colname);
            }
            else {
                this.cidx = dw.getSpecialFieldIndex(this.colname);
            }
            this.numdata = new HashMap<>();
        }


        @Override
        public void readLine(String key, DWARFileParser dw) {
            String ni = (this.specialfield)?dw.getSpecialFieldData(this.cidx):dw.getFieldData(this.cidx);
            ni = ni.trim();
            double[] dval = parseNumericData(ni);
            this.numdata.put(key,dval);
        }

        private double[] parseNumericData(String ni) {
            if (ni.isEmpty()) {
                //this.numdata.put(key, new double[0]);
                return new double[0];
            } else {
                String splits[] = ni.split(ConstantsDWAR.SEP_VALUE);
                double dval[] = new double[splits.length];
                for (int zi = 0; zi < splits.length; zi++) {
                    dval[zi] = Double.NaN;
                    try {
                        dval[zi] = Double.parseDouble(splits[zi]);
                    } catch (Exception ex) {
                    }
                }
                return dval;
                //this.numdata.put(key, dval);
            }
        }

        @Override
        public void postprocess(DWARFileParser dw) {
            this.mndp = new DefaultMultiNumericDataProvider(this.numdata);
        }

        @Override
        public NDataProvider.NMultiNumericDataProvider getDataProvider() {
            return this.mndp;
        }

        @Override
        public MultiNumericDataColumn getColumn() {
            return new MultiNumericDataColumn(this.colname);
        }
    }

    public static class StringColumnImporter implements DWARColumnImporter<NDataProvider.NStringDataProvider, StringColumn> {
        private String colname;
        private boolean specialfield;
        private int    cidx = -1;
        private DefaultStringDataProvider dsdp = null;
        Map<String,String> stringdata = new HashMap<>();
        public StringColumnImporter(String colname, boolean specialfield) {
            this.colname = colname;
            this.specialfield = specialfield;
        }

        @Override
        public boolean canParse(String data) {
            return true;
        }

        @Override
        public void prepare(DWARFileParser dw) {
            if(!this.specialfield) {
                this.cidx = dw.getFieldIndex(this.colname);
            }
            else {
                this.cidx = dw.getSpecialFieldIndex(this.colname);
            }
            this.stringdata = new HashMap<>();
        }

        @Override
        public void readLine(String key, DWARFileParser dw) {
            String ni = (this.specialfield)?dw.getSpecialFieldData(this.cidx):dw.getFieldData(this.cidx);
            this.stringdata.put(key,ni);
        }

        @Override
        public void postprocess(DWARFileParser dw) {
            this.dsdp= new DefaultStringDataProvider(this.stringdata);
        }

        @Override
        public NDataProvider.NStringDataProvider getDataProvider() {
            return this.dsdp;
        }

        @Override
        public StringColumn getColumn() {
            return new StringColumn(this.colname);
        }
    }



    private List<DWARColumnImporter> importers;
    private String pathDWAR;

    public DWARSimpleWrapper(String file, List<DWARColumnImporter> columns) {
        this.pathDWAR = file;
        this.importers = columns;
    }


    private NexusTableModel ntm = new NexusTableModel();
    public void loadFile() {
        this.ntm = new NexusTableModel();
        DWARFileParser dwarIn = new DWARFileParser(pathDWAR);
        String[] fields = dwarIn.getFieldNames();
        Map<String, DWARFileParser.SpecialField> sfields = dwarIn.getSpecialFieldMap();

        //while(dwarIn.next()) {
            //dwarIn.getFieldData();
        //}

        for(DWARColumnImporter ci : this.importers) {
            ci.prepare(dwarIn);
        }
        List<String> all_rowkeys = new ArrayList<>();
        int zi=0;
        while(dwarIn.next()) {
            String ki = ""+zi;
            all_rowkeys.add(ki);
            for(DWARColumnImporter ci : this.importers) {
                ci.readLine(ki,dwarIn);
            }
            zi++;
        }
        for(DWARColumnImporter ci : this.importers) {
            ci.postprocess(dwarIn);
        }

        this.ntm.setAllRows(all_rowkeys);
        for(int za=0;za<importers.size();za++) {
            this.ntm.setNexusColumnWithDataProvider(za, Pair.of(importers.get(za).getColumn(),importers.get(za).getDataProvider()));
        }
    }

    public NexusTableModel getNexusTableModel() {
        return this.ntm;
    }


}
