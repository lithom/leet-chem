package tech.molecules.leet.table.gui;

import com.actelion.research.chem.IsomericSmilesCreator;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.gui.editor.SwingEditorPanel;
import tech.molecules.leet.chem.ChemUtils;

import javax.swing.*;
import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.StringSelection;
import java.awt.datatransfer.UnsupportedFlavorException;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.PropertyChangeListener;
import java.io.IOException;

public class JExtendedEditorPanel extends JPanel  {

    private SwingEditorPanel mEditorPanel;

    private JPanel     mTop;
    private JPanel     mTop_Right;

    private JMenuBar   jmb_Main;
    private JMenu      jm_Tools;

    public JExtendedEditorPanel() {
        reinit();
    }

    private void reinit() {
        this.removeAll();

        StereoMolecule mi = new StereoMolecule();
        mi.setFragment(true);
        this.mEditorPanel = new SwingEditorPanel(new StereoMolecule());

        this.setLayout(new BorderLayout());
        this.add(this.mEditorPanel, BorderLayout.CENTER);

        this.mTop = new JPanel();
        this.mTop_Right = new JPanel();

        this.mTop.setLayout(new BorderLayout());
        this.add(mTop,BorderLayout.NORTH);
        this.mTop.add(this.mTop_Right,BorderLayout.EAST);

        // init Menu
        initToolsMenu();
        this.mTop_Right.setLayout(new FlowLayout(FlowLayout.RIGHT));
        this.jmb_Main = new JMenuBar();
        this.mTop_Right.add(this.jmb_Main);
        this.jmb_Main.add(this.jm_Tools);
    }

    private void initToolsMenu() {
        this.jm_Tools = new JMenu("Tools");
        JMenuItem paste = new JMenuItem("Paste");
        JMenu export_expanded = new JMenu("Export..");
        JMenuItem copy_idcode  = new JMenuItem("Copy IDCode");
        JMenuItem copy_smiles  = new JMenuItem("Copy Smiles");


        this.jm_Tools.add(paste);
        this.jm_Tools.add(export_expanded);
        export_expanded.add(copy_idcode);
        export_expanded.add(copy_smiles);

        paste.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    String str_data = (String) Toolkit.getDefaultToolkit().getSystemClipboard().getData(DataFlavor.stringFlavor);
                    StereoMolecule m_parsed = ChemUtils.tryParseChemistry(str_data);
                    if(m_parsed!=null) {
                        //mEditorPanel.getDrawArea().getMolecule().addFragment(m_parsed,0,null);
                        //mEditorPanel.cleanStructure();
                        StereoMolecule mi = new StereoMolecule(mEditorPanel.getDrawArea().getMolecule());
                        mi.ensureHelperArrays(Molecule.cHelperCIP);
                        mi.addFragment(m_parsed,0,null);
                        mEditorPanel.getDrawArea().setMolecule(m_parsed);
                        mEditorPanel.cleanStructure();
                    }
                } catch (UnsupportedFlavorException ex) {
                    System.out.println("[WARN] exception");
                    //throw new RuntimeException(ex);
                } catch (IOException ex) {
                    System.out.println("[WARN] exception");
                    //throw new RuntimeException(ex);
                }

            }
        });

        copy_idcode.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                String idc = mEditorPanel.getDrawArea().getMolecule().getIDCode();
                StringSelection stringSelection = new StringSelection(idc);

                Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
                clipboard.setContents(stringSelection, null);
            }
        });
        copy_smiles.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                //String idc = mDrawPanel.getDrawArea().getMolecule().getIDCode();
                IsomericSmilesCreator isc = new IsomericSmilesCreator(mEditorPanel.getDrawArea().getMolecule());
                String smiles = isc.getSmiles();
                StringSelection stringSelection = new StringSelection(smiles);
                Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
                clipboard.setContents(stringSelection, null);
            }
        });
    }


    public SwingEditorPanel getSwingEditorPanel() {
        return this.mEditorPanel;
    }

    public JMenuBar getMenuBar() {
        return this.jmb_Main;
    }

    private class JExtendedDrawPanelBar {

    }

}
