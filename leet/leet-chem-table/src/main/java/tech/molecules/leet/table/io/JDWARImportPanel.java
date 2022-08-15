package tech.molecules.leet.table.io;

import com.actelion.research.chem.io.DWARFileParser;
import com.actelion.research.gui.VerticalFlowLayout;
import org.apache.commons.lang3.tuple.Pair;
import tech.molecules.leet.table.NColumn;
import tech.molecules.leet.table.NDataProvider;
import tech.molecules.leet.table.NexusTable;
import tech.molecules.leet.table.NexusTableModel;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class JDWARImportPanel extends JPanel {

    public static class ImportOption<U extends NDataProvider,T extends NColumn> {
        public final String name;
        public final DWARSimpleWrapper.DWARColumnImporter<U,T> importer;
        public ImportOption(String name, DWARSimpleWrapper.DWARColumnImporter<U,T> importer) {
            this.name = name;
            this.importer = importer;
        }

        public String toString() {
            return this.name;
        }
    }

    private String filepath;

    private List<ColumnImportPanel> importPanels = new ArrayList<>();
    //private ImportOption<NDataProvider,NColumn> NoImport = new ImportOption("No Import",null);

    public void autoconfigure_A() {

        for(ColumnImportPanel ci : this.importPanels){
            DWARFileParser ain = new DWARFileParser(this.filepath);
            int zn = 0;
            while(zn<50 && ain.next()) {
                String di = "";
                if(ci.specialField) {
                    di = ain.getSpecialFieldData( ain.getSpecialFieldIndex(ci.name));
                }
                else {
                    di = ain.getFieldData(ain.getFieldIndex(ci.name));
                }
                boolean can_parse_structure = new DWARSimpleWrapper.StructureColumnImporter(ci.name,ci.specialField).canParse(di);
                if(can_parse_structure) {
                    ci.jbImport.setSelectedIndex(2);
                    break;
                }
                boolean can_parse_numeric = new DWARSimpleWrapper.MultiNumericColumnImporter(ci.name,ci.specialField).canParse(di);
                if(can_parse_numeric) {
                    ci.jbImport.setSelectedIndex(1);
                    break;
                }
            }
            ain.close();
        }
    }
    public JDWARImportPanel(String filepath) {
        this.filepath = filepath;

        DWARFileParser in = new DWARFileParser(filepath);
        String[]     fields = in.getFieldNames();
        List<String> sfields = new ArrayList<>(in.getSpecialFieldMap().keySet());

        JScrollPane jsp = new JScrollPane();
        JPanel      ja  = new JPanel();
        jsp.setViewportView(ja);
        ja.setLayout(new VerticalFlowLayout());

        for(String fi : fields) {
            ColumnImportPanel ci = new ColumnImportPanel(fi,false);
            ja.add(ci);
            importPanels.add(ci);
        }
        for(String fi : sfields) {
            ColumnImportPanel ci = new ColumnImportPanel(fi,true);
            ja.add(ci);
            importPanels.add(ci);
        }
        this.setLayout(new BorderLayout());
        this.add(jsp,BorderLayout.CENTER);

        JButton jbLoad = new JButton("Import");
        this.add(jbLoad,BorderLayout.SOUTH);
        jbLoad.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //List<Pair<NDataProvider,NColumn>> nexusColumns = new ArrayList<>();
                List<DWARSimpleWrapper.DWARColumnImporter> importers = importPanels.stream().filter(fi -> fi.getImporter() !=null).map(fi
                        -> fi.getImporter() ).collect(Collectors.toList());
                DWARSimpleWrapper wrapper = new DWARSimpleWrapper(filepath,importers);
                wrapper.loadFile();
                NexusTableModel ntm = wrapper.getNexusTableModel();
                NexusTable nt = new NexusTable(ntm);
                JFrame fi = new JFrame();
                JScrollPane jspi = new JScrollPane(nt);
                jspi.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
                jspi.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
                nt.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);

                fi.getContentPane().setLayout(new BorderLayout());
                fi.getContentPane().add(jspi,BorderLayout.CENTER);
                fi.setSize(600,600);
                fi.setVisible(true);
            }
        });

        this.autoconfigure_A();
    }

    class ColumnImportPanel extends JPanel {
        String name;
        boolean specialField;
        public ColumnImportPanel(String name, boolean special) {
            this.name = name;
            this.specialField = special;
            this.initImportPanel();
        }
        JLabel jlName;
        JComboBox<ImportOption> jbImport;
        private void initImportPanel() {
            this.setLayout(new FlowLayout());
            this.jlName = new JLabel(this.name);
            this.jbImport = new JComboBox<>();
            jbImport.addItem(new ImportOption("String",new DWARSimpleWrapper.StringColumnImporter(this.name,this.specialField)));
            jbImport.addItem(new ImportOption("Numeric",new DWARSimpleWrapper.MultiNumericColumnImporter(this.name,this.specialField)));
            jbImport.addItem(new ImportOption("Structure",new DWARSimpleWrapper.StructureColumnImporter(this.name,this.specialField)));
            jbImport.addItem(new ImportOption("No import", null));
            this.add(jlName);
            this.add(jbImport);
        }

        public DWARSimpleWrapper.DWARColumnImporter getImporter() {
            return ((ImportOption)this.jbImport.getSelectedItem()).importer;
        }
    }

    public static void showImportDialog() {
    }

    public static void main(String args[]) {
        JFrame fi = new JFrame();
        fi.getContentPane();
        fi.getContentPane().setLayout(new BorderLayout());
        //JDWARImportPanel ipp = new JDWARImportPanel("C:\\datasets\\test_set_mcs_indole.dwar");//screening_libs_raw.dwar");
        //JDWARImportPanel ipp = new JDWARImportPanel("C:\\datasets\\CM_Available_Compounds_a.dwar");//screening_libs_raw.dwar");
        JDWARImportPanel ipp = new JDWARImportPanel("C:\\datasets\\dwar\\cfbproject_a.dwar");//screening_libs_raw.dwar");
        fi.getContentPane().add(ipp,BorderLayout.CENTER);
        fi.setSize(600,600);
        fi.setVisible(true);
    }


}
