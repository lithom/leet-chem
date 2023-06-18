package tech.molecules.leet.gui.chem.editor;

import com.actelion.research.chem.Molecule;
import com.actelion.research.chem.StereoMolecule;
import com.actelion.research.gui.editor.SwingEditorPanel;
import tech.molecules.leet.chem.ChemUtils;
import tech.molecules.leet.chem.sar.SARDecompositionInstruction2;
import tech.molecules.leet.chem.sar.SARElement;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SARDecompositionEditor extends JPanel {

    private SwingEditorPanel editor;

    public SARDecompositionEditor() {
        reinitGUI();
        initListeners();
    }

    private void reinitGUI() {
        this.removeAll();
        this.setLayout(new BorderLayout());
        this.editor = new SwingEditorPanel(new StereoMolecule());
        this.add(this.editor,BorderLayout.CENTER);
        editor.getDrawArea().setAllowQueryFeatures(true);

        this.editor.setFocusable(true);
        this.editor.setRequestFocusEnabled(true);
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





    public static void main(String args[]) {
        JFrame fi = new JFrame();
        fi.setSize(600,600);

        SARDecompositionEditor editor = new SARDecompositionEditor();
        editor.setFocusable(true);
        editor.setRequestFocusEnabled(true);
        fi.getContentPane().setLayout(new BorderLayout());
        fi.getContentPane().add(editor,BorderLayout.CENTER);
        fi.setVisible(true);
        SwingUtilities.updateComponentTreeUI(fi);
    }





}
