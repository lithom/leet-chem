package tech.molecules.leet.chem.sar;

import com.actelion.research.chem.StereoMolecule;

import java.util.ArrayList;
import java.util.List;

public class SimpleMultiFragment {

    private List<SimpleSARElement> elements;

    public SimpleMultiFragment(List<SimpleSARElement> elements) {
        this.elements = elements;
    }

    public SimpleMultiFragment() {
        SimpleSARElement ea = new SimpleSARElement(new StereoMolecule());
        List<SimpleSARElement> sel = new ArrayList<>();
        sel.add(ea);
        this.elements = sel;
    }

    public List<SimpleSARElement> getElements() {
        return elements;
    }

    private List<SimpleMultiFragment.MultiFragmentListener> elementListeners = new ArrayList<>();

    public void addMultiFragmentElement(SimpleSARElement element) {
        elements.add(element);
        fireMultiFragmentElementAdded(element);
    }

    public void removeMultiFragmentElement(int index) {
        elements.remove(index);
        fireMultiFragmentElementRemoved(index);
    }

    public void setFragmentEdited(SimpleSARElement element) {
        this.fireSARElementSelected(element);
    }

    public void addMultiFragmentElementListener(MultiFragmentListener listener) {
        elementListeners.add(listener);
    }

    public void removeMultiFragmentElementListener(MultiFragmentListener listener) {
        elementListeners.remove(listener);
    }

    private void fireMultiFragmentElementAdded(SimpleSARElement element) {
        for (MultiFragmentListener listener : elementListeners) {
            listener.onMultiFragmentElementAdded(element);
        }
    }

    private void fireMultiFragmentElementRemoved(int index) {
        for (MultiFragmentListener listener : elementListeners) {
            listener.onMultiFragmentElementRemoved(index);
        }
    }

    private void fireSARElementSelected(SimpleSARElement element) {
        for (MultiFragmentListener listener : elementListeners) {
            listener.onSARElementSelected(element);
        }
    }

    public interface MultiFragmentListener {
        void onMultiFragmentElementAdded(SimpleSARElement element);
        void onMultiFragmentElementRemoved(int index);
        void onSARElementSelected(SimpleSARElement element);
    }

}
