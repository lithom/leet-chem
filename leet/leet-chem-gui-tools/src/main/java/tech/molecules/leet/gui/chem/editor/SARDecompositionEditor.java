package tech.molecules.leet.gui.chem.editor;

import com.actelion.research.chem.Canonizer;
import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.gui.editor.GenericEditorArea;
import com.actelion.research.gui.editor.SwingEditorPanel;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.StructureWithID;
import tech.molecules.leet.chem.sar.*;
import tech.molecules.leet.gui.chem.action.LoadStructureWithIDFromDWARAction;

import javax.swing.*;
import java.awt.*;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

public class SARDecompositionEditor extends JPanel {

    private SimpleSARDecompositionModel decompositionModel = null;

    private List<StructureWithID> structures = new ArrayList<>();

    private SwingEditorPanel editor;

    public SARDecompositionEditor() {
        reinitGUI();
        initListeners();
    }

    public JPanel getThisPanel() {return this;}

    private void reinitGUI() {

        this.removeAll();
        this.setLayout(new BorderLayout());
        StereoMolecule ma = new StereoMolecule();
        ma.setFragment(true);
        this.editor = new SwingEditorPanel(ma);
        this.add(this.editor,BorderLayout.CENTER);
        editor.getDrawArea().setAllowQueryFeatures(true);
        //StereoMolecule sa = new StereoMolecule();
        //sa.setFragment(true);
        editor.getDrawArea().setMolecule(ma);
        editor.getDrawArea().setAllowQueryFeatures(true);
        editor.getDrawArea().setMolecule(ma);

        this.editor.setFocusable(true);
        this.editor.setRequestFocusEnabled(true);



        JPanel topbar = new JPanel(); topbar.setLayout(new FlowLayout(FlowLayout.LEFT));
        JButton jbExport = new JButton("Export");
        JButton jbSetLabel = new JButton("Set Label");
        JButton jbLoad     = new JButton("Load Structures..");
        JButton jbAnalyze   = new JButton("Analyze..");
        topbar.add(jbSetLabel);
        topbar.add(jbExport);
        topbar.add(jbLoad);
        topbar.add(jbAnalyze);
        jbExport.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                Canonizer ca = new Canonizer(editor.getDrawArea().getMolecule(),Canonizer.ENCODE_ATOM_CUSTOM_LABELS);
                String theString = ca.getIDCode();
                StringSelection selection = new StringSelection(theString);
                Clipboard clipboard = Toolkit.getDefaultToolkit().getSystemClipboard();
                clipboard.setContents(selection, selection);
            }
        });
        jbSetLabel.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {

                BlockingDialog db = new BlockingDialog((Frame) SwingUtilities.getWindowAncestor(getThisPanel()) ,"");
                db.setVisible(true);
                for(int zi=0;zi<editor.getDrawArea().getMolecule().getAtoms();zi++) {
                    if( editor.getDrawArea().getMolecule().isSelectedAtom(zi) ) {
                        editor.getDrawArea().getMolecule().setAtomCustomLabel(zi, db.getEnteredText());
                    }
                }
                editor.getDrawArea().setMolecule(editor.getDrawArea().getMolecule());
                editor.getSwingDrawArea().repaint();
            }
        });
        jbLoad.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                LoadStructureWithIDFromDWARAction loadAction = new LoadStructureWithIDFromDWARAction();
                loadAction.actionPerformed(e);
                structures = loadAction.getStructures();
            }
        });
        jbAnalyze.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                SimpleSARSeries sis = new SimpleSARSeries("test", new SimpleMultiSynthonStructure(editor.getDrawArea().getMolecule()));
                decompositionModel = new SimpleSARDecompositionModel(Collections.singletonList(sis));
                Future fi = decompositionModel.addCompounds(structures.stream().map(xi -> xi.structure[0]).collect(Collectors.toList()));
                Thread ti = new Thread() {
                    @Override
                    public void run() {
                        try {
                            fi.get();
                        } catch (InterruptedException ex) {
                            ex.printStackTrace();
                            //throw new RuntimeException(ex);
                        } catch (ExecutionException ex) {
                            ex.printStackTrace();
                            //throw new RuntimeException(ex);
                        }
                        fireDecompositionChanged(decompositionModel);
                    }
                };
                ti.start();
            }
        });


        this.add(topbar,BorderLayout.NORTH);


        SwingUtilities.updateComponentTreeUI(this);
    }

    private void initListeners() {
        this.editor.getSwingDrawArea().addKeyListener(new KeyListener() {
            @Override
            public void keyTyped(KeyEvent e) {

            }

            @Override
            public void keyPressed(KeyEvent e) {
                if( editor.getDrawArea().getHiliteAtom() >= 0) {
                    System.out.println("ctrl: " + e.isControlDown() + " c: " + e.getKeyCode());
                    if (e.isControlDown() && e.getKeyCode() == 69) {
                        actionOnAtom(editor.getDrawArea().getHiliteAtom());
                        //List<Integer> si = getSelectedAtoms(editor.getDrawArea().getMolecule());
                        //if (si.size() == 1) {
                        //    System.out.println("action: " + si.get(0));
                        //    actionOnAtom(si.get(0));
                        //}
                    }
                }
            }

            @Override
            public void keyReleased(KeyEvent e) {

            }
        });
    }

    private int label_cnt = 0;
    private void actionOnAtom(int atom) {
        StereoMolecule mol = editor.getDrawArea().getMolecule();
        if (mol.getAtomCustomLabel(atom) != null && !mol.getAtomCustomLabel(atom).equals("")) {
            mol.setAtomCustomLabel(atom, (String)null);
            this.editor.getDrawArea().setMolecule(mol);
        } else {
            mol.setAtomCustomLabel(atom, "LABEL_"+label_cnt);
            label_cnt++;
            this.editor.getDrawArea().setMolecule(mol);
        }
        this.editor.getSwingDrawArea().repaint();
        BlockingDialog db = new BlockingDialog((Frame) SwingUtilities.getWindowAncestor(this) ,mol.getAtomCustomLabel(atom));
        db.setVisible(true);
        mol.setAtomCustomLabel(atom, db.getEnteredText());
        this.editor.getDrawArea().setMolecule(mol);
        this.editor.getDrawArea().setAllowQueryFeatures(true);

        computeTestDecomposition();
    }



    private void computeTestDecomposition() {
        String idcode = "ekhRHH@F@fao@@cIEDhThdeDYdlhdbJIEGHjfmcje@@`Xjj@BB@@@";
        StereoMolecule ma = ChemUtils.parseIDCode(idcode);

        // 1. create decomp :)
        StereoMolecule mdc = new StereoMolecule(editor.getDrawArea().getMolecule());
        mdc.ensureHelperArrays(Molecule.cHelperCIP);
        StereoMolecule mf[] = mdc.getFragments();

        List<SARElement> sar_elements = new ArrayList<>();
        for(StereoMolecule mfi : mf) {
            mfi.ensureHelperArrays(Molecule.cHelperCIP);
            SARElement si = new SARElement(mfi);
            sar_elements.add(si);
        }
        SARDecompositionInstruction2.matchSARElements(sar_elements, Collections.singletonList(ma));
    }

    public static List<Integer> getSelectedAtoms(StereoMolecule mi) {
        List<Integer> si = new ArrayList<>();
        for(int zi=0;zi<mi.getAtoms();zi++) {
            if(mi.isSelectedAtom(zi)) {si.add(zi);}
        }
        return si;
    }


    public class BlockingDialog extends JDialog {
        private JTextField textField;

        public BlockingDialog(Frame parent, String initialText) {
            super(parent, "Atom Label Dialog", true);

            textField = new JTextField(initialText);

            JButton confirmButton = new JButton("Confirm");
            confirmButton.addActionListener(new ActionListener() {
                @Override
                public void actionPerformed(ActionEvent e) {
                    dispose(); // Close the dialog when the button is clicked
                }
            });

            JPanel panel = new JPanel(new GridLayout(2, 1));
            panel.add(textField);
            panel.add(confirmButton);

            setContentPane(panel);
            pack();
            setLocationRelativeTo(parent);
        }

        public String getEnteredText() {
            return textField.getText();
        }
    }



    public static interface DecompositionListener {
        public void decompositionChanged(SimpleSARDecompositionModel decompModel);
    }

    private List<DecompositionListener> listeners = new ArrayList<>();
    public void addDecompositionListener(DecompositionListener li) {this.listeners.add(li);}
    public boolean removeDecompositionListener(DecompositionListener li) {return this.listeners.remove(li);}
    private void fireDecompositionChanged(SimpleSARDecompositionModel decompModel) {
        for(DecompositionListener li : listeners) { li.decompositionChanged(decompModel); }
    }




    public static void main(String args[]) {
        JFrame fi = new JFrame();
        fi.setSize(600,600);

        if(true) {
            SARDecompositionEditor editor = new SARDecompositionEditor();
            editor.setFocusable(true);
            editor.setRequestFocusEnabled(true);
            fi.getContentPane().setLayout(new BorderLayout());
            fi.getContentPane().add(editor, BorderLayout.CENTER);
            fi.setVisible(true);
            SwingUtilities.updateComponentTreeUI(fi);
        }
        if(false) {
            SwingEditorPanel editor = new SwingEditorPanel(new StereoMolecule());
            editor.setFocusable(true);
            editor.setRequestFocusEnabled(true);
            editor.getDrawArea().setAllowQueryFeatures(true);
            StereoMolecule sa = new StereoMolecule();
            sa.setFragment(true);
            editor.getDrawArea().setMolecule(sa);
            fi.getContentPane().setLayout(new BorderLayout());
            fi.getContentPane().add(editor, BorderLayout.CENTER);
            fi.setVisible(true);
            SwingUtilities.updateComponentTreeUI(fi);
        }
    }





}
