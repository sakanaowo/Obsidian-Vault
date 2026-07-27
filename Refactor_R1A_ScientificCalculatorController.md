# Refactor R1A — ScientificCalculatorController

Tài liệu hướng dẫn tách toàn bộ logic Scientific khỏi `MainActivity` sang `ScientificCalculatorController`.


## Mục tiêu

Ở bước này:

```text
Tạo 1 file mới
Không sửa MainActivity
Không xóa code cũ
Không thay đổi hành vi app
```

Controller mới chưa được nối vào `MainActivity`, vì vậy app vẫn chạy bằng implementation 8.7 hiện tại. Mục tiêu của R1A là tạo một file controller độc lập và kiểm tra compile trước khi chuyển listener/state khỏi Activity.

## Đường dẫn package

Tạo package:

```text
app/src/main/java/com/example/calculator/ui/controller
```

Package đầy đủ:

```text
com.example.calculator.ui.controller
```

Tạo file:

```text
app/src/main/java/com/example/calculator/ui/controller/
ScientificCalculatorController.java
```

## Nội dung hoàn chỉnh của `ScientificCalculatorController.java`

```java
package com.example.calculator.ui.controller;

import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.example.calculator.R;
import com.example.calculator.engine.AngleMode;
import com.example.calculator.engine.ScientificEngine;
import com.example.calculator.history.HistoryManager;

/**
 * Điều phối toàn bộ giao diện Scientific Calculator.
 *
 * Trách nhiệm:
 *
 * - Bind Scientific Views
 * - Nhận button click
 * - Quản lý Scientific input state
 * - Gọi ScientificEngine
 * - Render expression/result
 * - Lưu History khi calculation thành công
 *
 * Không thực hiện công thức toán học trực tiếp.
 */
public final class ScientificCalculatorController {

    // =========================================================
    // DEPENDENCIES
    // =========================================================

    private final View rootView;
    private final ScientificEngine scientificEngine;
    private final HistoryManager historyManager;


    // =========================================================
    // DISPLAY VIEWS
    // =========================================================

    private TextView tvSciExpression;
    private TextView tvSciResult;
    private TextView tvSciAngleIndicator;
    private Button btnSciAngleMode;


    // =========================================================
    // INPUT STATE
    // =========================================================

    private String scientificInput;
    private String scientificExpression;
    private String scientificResult;
    private boolean scientificJustEvaluated;


    // =========================================================
    // PENDING BINARY OPERATION
    // =========================================================

    private ScientificPendingOperation scientificPendingOperation;
    private double scientificPendingFirstValue;
    private String scientificPendingFirstText;


    private enum ScientificPendingOperation {
        NONE,
        POWER,
        Y_ROOT
    }


    private interface ScientificUnaryOperation {

        ScientificEngine.ScientificResult calculate(
                double value
        );
    }


    // =========================================================
    // CONSTRUCTOR
    // =========================================================

    public ScientificCalculatorController(
            View rootView,
            ScientificEngine scientificEngine,
            HistoryManager historyManager
    ) {

        if (rootView == null) {
            throw new IllegalArgumentException(
                    "Scientific root view cannot be null."
            );
        }

        if (scientificEngine == null) {
            throw new IllegalArgumentException(
                    "ScientificEngine cannot be null."
            );
        }

        this.rootView = rootView;
        this.scientificEngine = scientificEngine;
        this.historyManager = historyManager;

        resetScientificState();
    }


    // =========================================================
    // PUBLIC API
    // =========================================================

    public void setup() {

        bindViews();
        setupScientificDigitButtons();
        setupScientificOperatorButtons();
        setupScientificFunctionButtons();
        renderScientificState();
    }


    public void clear() {

        resetScientificState();
        renderScientificState();
    }


    // =========================================================
    // BIND VIEWS
    // =========================================================

    private void bindViews() {

        tvSciExpression =
                findScientificView(
                        R.id.tvSciExpression
                );

        tvSciResult =
                findScientificView(
                        R.id.tvSciResult
                );

        tvSciAngleIndicator =
                findScientificView(
                        R.id.tvSciAngleIndicator
                );

        btnSciAngleMode =
                findScientificView(
                        R.id.btnSciAngleMode
                );
    }


    private <T extends View> T findScientificView(
            int viewId
    ) {

        return rootView.findViewById(viewId);
    }


    // =========================================================
    // DIGIT BUTTONS
    // =========================================================

    private void setupScientificDigitButtons() {

        bindScientificDigitButton(R.id.btnSci0, "0");
        bindScientificDigitButton(R.id.btnSci1, "1");
        bindScientificDigitButton(R.id.btnSci2, "2");
        bindScientificDigitButton(R.id.btnSci3, "3");
        bindScientificDigitButton(R.id.btnSci4, "4");
        bindScientificDigitButton(R.id.btnSci5, "5");
        bindScientificDigitButton(R.id.btnSci6, "6");
        bindScientificDigitButton(R.id.btnSci7, "7");
        bindScientificDigitButton(R.id.btnSci8, "8");
        bindScientificDigitButton(R.id.btnSci9, "9");
    }


    private void bindScientificDigitButton(
            int buttonId,
            String digit
    ) {

        findScientificView(
                buttonId
        ).setOnClickListener(view ->
                appendScientificDigit(digit)
        );
    }


    private void appendScientificDigit(
            String digit
    ) {

        if (digit == null
                || !digit.matches("[0-9]")) {

            return;
        }

        prepareScientificForValueInput();

        if ("0".equals(scientificInput)) {
            scientificInput = digit;
        } else {
            scientificInput += digit;
        }

        scientificJustEvaluated = false;

        if (scientificPendingOperation
                == ScientificPendingOperation.NONE) {

            scientificExpression = "";
        }

        renderScientificState();
    }


    // =========================================================
    // DECIMAL
    // =========================================================

    private void appendScientificDecimal() {

        prepareScientificForValueInput();

        String currentNumber =
                getScientificCurrentNumberToken();

        if (currentNumber.contains(".")) {
            return;
        }

        if (currentNumber.isEmpty()
                || "-".equals(currentNumber)) {

            scientificInput += "0.";

        } else {

            scientificInput += ".";
        }

        scientificJustEvaluated = false;

        if (scientificPendingOperation
                == ScientificPendingOperation.NONE) {

            scientificExpression = "";
        }

        renderScientificState();
    }


    private String getScientificCurrentNumberToken() {

        if (scientificInput == null
                || scientificInput.isEmpty()) {

            return "";
        }

        int index =
                scientificInput.length() - 1;

        while (index >= 0) {

            char character =
                    scientificInput.charAt(index);

            if (!Character.isDigit(character)
                    && character != '.'
                    && character != '-') {

                break;
            }

            index--;
        }

        return scientificInput.substring(index + 1);
    }


    // =========================================================
    // ARITHMETIC OPERATORS
    // =========================================================

    private void setupScientificOperatorButtons() {

        bindScientificOperatorButton(
                R.id.btnSciAdd,
                "+"
        );

        bindScientificOperatorButton(
                R.id.btnSciSubtract,
                "−"
        );

        bindScientificOperatorButton(
                R.id.btnSciMultiply,
                "×"
        );

        bindScientificOperatorButton(
                R.id.btnSciDivide,
                "÷"
        );
    }


    private void bindScientificOperatorButton(
            int buttonId,
            String operator
    ) {

        findScientificView(
                buttonId
        ).setOnClickListener(view ->
                appendScientificOperator(operator)
        );
    }


    private void appendScientificOperator(
            String operator
    ) {

        if (operator == null
                || operator.trim().isEmpty()) {

            return;
        }

        if (scientificJustEvaluated
                && scientificPendingOperation
                == ScientificPendingOperation.NONE) {

            scientificInput = scientificResult;
            scientificExpression = "";
            scientificJustEvaluated = false;
        }

        if (scientificInput == null
                || scientificInput.trim().isEmpty()) {

            if (scientificResult == null
                    || scientificResult.trim().isEmpty()
                    || "Error".equals(scientificResult)) {

                return;
            }

            scientificInput = scientificResult;
        }

        scientificInput =
                removeTrailingScientificOperator(
                        scientificInput
                );

        scientificInput +=
                " " + operator + " ";

        scientificJustEvaluated = false;

        renderScientificState();
    }


    private String removeTrailingScientificOperator(
            String expression
    ) {

        if (expression == null) {
            return "";
        }

        String updated = expression.trim();

        while (!updated.isEmpty()) {

            char last =
                    updated.charAt(
                            updated.length() - 1
                    );

            if (last == '+'
                    || last == '−'
                    || last == '-'
                    || last == '×'
                    || last == '÷'
                    || last == '*'
                    || last == '/'
                    || last == '^') {

                updated =
                        updated.substring(
                                0,
                                updated.length() - 1
                        ).trim();

            } else {

                break;
            }
        }

        return updated;
    }


    // =========================================================
    // FUNCTION BUTTONS
    // =========================================================

    private void setupScientificFunctionButtons() {

        findScientificView(
                R.id.btnSciDecimal
        ).setOnClickListener(view ->
                appendScientificDecimal()
        );

        findScientificView(
                R.id.btnSciEquals
        ).setOnClickListener(view ->
                calculateScientificExpression()
        );

        findScientificView(
                R.id.btnSciClear
        ).setOnClickListener(view ->
                clear()
        );

        findScientificView(
                R.id.btnSciBackspace
        ).setOnClickListener(view ->
                scientificBackspace()
        );

        findScientificView(
                R.id.btnSciToggleSign
        ).setOnClickListener(view ->
                toggleScientificSign()
        );

        findScientificView(
                R.id.btnSciPercent
        ).setOnClickListener(view ->
                applyScientificPercent()
        );

        findScientificView(
                R.id.btnSciOpenParen
        ).setOnClickListener(view ->
                appendScientificOpenParenthesis()
        );

        findScientificView(
                R.id.btnSciCloseParen
        ).setOnClickListener(view ->
                appendScientificCloseParenthesis()
        );

        findScientificView(
                R.id.btnSciSquare
        ).setOnClickListener(view ->
                performScientificUnary(
                        scientificEngine::square
                )
        );

        findScientificView(
                R.id.btnSciCube
        ).setOnClickListener(view ->
                performScientificUnary(
                        scientificEngine::cube
                )
        );

        findScientificView(
                R.id.btnSciPower
        ).setOnClickListener(view ->
                beginScientificPendingOperation(
                        ScientificPendingOperation.POWER
                )
        );

        findScientificView(
                R.id.btnSciReciprocal
        ).setOnClickListener(view ->
                performScientificUnary(
                        scientificEngine::reciprocal
                )
        );

        findScientificView(
                R.id.btnSciSqrt
        ).setOnClickListener(view ->
                performScientificUnary(
                        scientificEngine::squareRoot
                )
        );

        findScientificView(
                R.id.btnSciCbrt
        ).setOnClickListener(view ->
                performScientificUnary(
                        scientificEngine::cubeRoot
                )
        );

        findScientificView(
                R.id.btnSciYRoot
        ).setOnClickListener(view ->
                beginScientificPendingOperation(
                        ScientificPendingOperation.Y_ROOT
                )
        );

        findScientificView(
                R.id.btnSciFactorial
        ).setOnClickListener(view ->
                performScientificUnary(
                        scientificEngine::factorial
                )
        );

        findScientificView(
                R.id.btnSciSin
        ).setOnClickListener(view ->
                performScientificUnary(
                        scientificEngine::sin
                )
        );

        findScientificView(
                R.id.btnSciCos
        ).setOnClickListener(view ->
                performScientificUnary(
                        scientificEngine::cos
                )
        );

        findScientificView(
                R.id.btnSciTan
        ).setOnClickListener(view ->
                performScientificUnary(
                        scientificEngine::tan
                )
        );

        findScientificView(
                R.id.btnSciLn
        ).setOnClickListener(view ->
                performScientificUnary(
                        scientificEngine::naturalLog
                )
        );

        findScientificView(
                R.id.btnSciLog10
        ).setOnClickListener(view ->
                performScientificUnary(
                        scientificEngine::log10
                )
        );

        findScientificView(
                R.id.btnSciE
        ).setOnClickListener(view ->
                appendScientificConstant("e")
        );

        findScientificView(
                R.id.btnSciPi
        ).setOnClickListener(view ->
                appendScientificConstant("π")
        );

        findScientificView(
                R.id.btnSciExp
        ).setOnClickListener(view ->
                performScientificUnary(
                        scientificEngine::exponential
                )
        );

        findScientificView(
                R.id.btnSciTenPower
        ).setOnClickListener(view ->
                performScientificUnary(
                        scientificEngine::tenPower
                )
        );

        btnSciAngleMode.setOnClickListener(view -> {

            scientificEngine.toggleAngleMode();
            renderScientificState();
        });
    }


    // =========================================================
    // UNARY OPERATIONS
    // =========================================================

    private void performScientificUnary(
            ScientificUnaryOperation operation
    ) {

        if (operation == null) {
            return;
        }

        ScientificEngine.ScientificResult
                operandResult =
                evaluateScientificCurrentInput();

        if (!operandResult.isSuccess()) {

            applyScientificResult(
                    operandResult,
                    false
            );

            return;
        }

        ScientificEngine.ScientificResult result =
                operation.calculate(
                        operandResult.getValue()
                );

        boolean shouldSaveHistory =
                scientificPendingOperation
                        == ScientificPendingOperation.NONE;

        applyScientificResult(
                result,
                shouldSaveHistory
        );
    }


    private ScientificEngine.ScientificResult
    evaluateScientificCurrentInput() {

        String source =
                scientificInput == null
                        ? ""
                        : scientificInput.trim();

        if (source.isEmpty()
                && scientificResult != null
                && !"Error".equals(scientificResult)) {

            source = scientificResult;
        }

        return scientificEngine
                .evaluateExpression(source);
    }


    // =========================================================
    // POWER / Y ROOT
    // =========================================================

    private void beginScientificPendingOperation(
            ScientificPendingOperation operation
    ) {

        if (operation == null
                || operation
                == ScientificPendingOperation.NONE) {

            return;
        }

        ScientificEngine.ScientificResult
                firstOperandResult =
                evaluateScientificCurrentInput();

        if (!firstOperandResult.isSuccess()) {

            applyScientificResult(
                    firstOperandResult,
                    false
            );

            return;
        }

        scientificPendingOperation = operation;

        scientificPendingFirstValue =
                firstOperandResult.getValue();

        scientificPendingFirstText =
                firstOperandResult.getFormattedValue();

        scientificInput = "";
        scientificExpression = "";
        scientificResult = "0";
        scientificJustEvaluated = false;

        renderScientificState();
    }


    private void clearScientificPendingOperation() {

        scientificPendingOperation =
                ScientificPendingOperation.NONE;

        scientificPendingFirstValue = 0.0;
        scientificPendingFirstText = "";
    }


    // =========================================================
    // EQUALS
    // =========================================================

    private void calculateScientificExpression() {

        if (scientificJustEvaluated
                && scientificPendingOperation
                == ScientificPendingOperation.NONE) {

            return;
        }

        ScientificEngine.ScientificResult result;

        if (scientificPendingOperation
                == ScientificPendingOperation.NONE) {

            result =
                    scientificEngine
                            .evaluateExpression(
                                    scientificInput
                            );

        } else {

            ScientificEngine.ScientificResult
                    secondOperandResult =
                    evaluateScientificCurrentInput();

            if (!secondOperandResult.isSuccess()) {

                applyScientificResult(
                        secondOperandResult,
                        false
                );

                clearScientificPendingOperation();

                return;
            }

            double secondValue =
                    secondOperandResult.getValue();

            if (scientificPendingOperation
                    == ScientificPendingOperation.POWER) {

                result =
                        scientificEngine.power(
                                scientificPendingFirstValue,
                                secondValue
                        );

            } else {

                result =
                        scientificEngine.yRoot(
                                scientificPendingFirstValue,
                                secondValue
                        );
            }

            clearScientificPendingOperation();
        }

        applyScientificResult(
                result,
                true
        );
    }


    // =========================================================
    // RESULT + HISTORY
    // =========================================================

    private void applyScientificResult(
            ScientificEngine.ScientificResult result,
            boolean saveHistory
    ) {

        if (result == null) {
            return;
        }

        scientificExpression =
                result.getExpression() == null
                        ? ""
                        : result.getExpression();

        scientificResult =
                result.getFormattedValue() == null
                        ? "Error"
                        : result.getFormattedValue();

        if (result.isSuccess()) {

            scientificInput =
                    result.getFormattedValue();

            scientificJustEvaluated = true;

            if (saveHistory) {
                saveScientificHistory(result);
            }

        } else {

            scientificInput = "";
            scientificResult = "Error";

            if (result.getErrorMessage() != null
                    && !result
                    .getErrorMessage()
                    .trim()
                    .isEmpty()) {

                scientificExpression =
                        result.getErrorMessage();
            }

            scientificJustEvaluated = true;

            clearScientificPendingOperation();
        }

        renderScientificState();
    }


    private void saveScientificHistory(
            ScientificEngine.ScientificResult result
    ) {

        if (historyManager == null
                || result == null
                || !result.isSuccess()) {

            return;
        }

        String expression =
                result.getExpression();

        String formattedResult =
                result.getFormattedValue();

        if (expression == null
                || expression.trim().isEmpty()) {

            return;
        }

        if (formattedResult == null
                || formattedResult.trim().isEmpty()
                || "Error".equals(formattedResult)) {

            return;
        }

        historyManager.add(
                expression,
                formattedResult
        );
    }


    // =========================================================
    // CONSTANTS
    // =========================================================

    private void appendScientificConstant(
            String constant
    ) {

        if (constant == null
                || constant.trim().isEmpty()) {

            return;
        }

        prepareScientificForValueInput();

        if (scientificInput != null
                && !scientificInput.isEmpty()) {

            char last =
                    scientificInput.charAt(
                            scientificInput.length() - 1
                    );

            if (Character.isDigit(last)
                    || last == ')'
                    || last == 'π'
                    || last == 'e') {

                scientificInput += " × ";
            }
        }

        scientificInput += constant;
        scientificJustEvaluated = false;

        if (scientificPendingOperation
                == ScientificPendingOperation.NONE) {

            scientificExpression = "";
        }

        renderScientificState();
    }


    // =========================================================
    // PARENTHESES
    // =========================================================

    private void appendScientificOpenParenthesis() {

        prepareScientificForValueInput();

        if (scientificInput != null
                && !scientificInput.isEmpty()) {

            char last =
                    scientificInput.charAt(
                            scientificInput.length() - 1
                    );

            if (Character.isDigit(last)
                    || last == ')'
                    || last == 'π'
                    || last == 'e') {

                scientificInput += " × ";
            }
        }

        scientificInput += "(";
        scientificJustEvaluated = false;

        renderScientificState();
    }


    private void appendScientificCloseParenthesis() {

        if (scientificInput == null
                || scientificInput.trim().isEmpty()) {

            return;
        }

        int openCount =
                countCharacter(
                        scientificInput,
                        '('
                );

        int closeCount =
                countCharacter(
                        scientificInput,
                        ')'
                );

        if (closeCount >= openCount) {
            return;
        }

        char last =
                scientificInput.charAt(
                        scientificInput.length() - 1
                );

        if (last == '('
                || last == '+'
                || last == '−'
                || last == '-'
                || last == '×'
                || last == '÷'
                || last == '^'
                || Character.isWhitespace(last)) {

            return;
        }

        scientificInput += ")";
        scientificJustEvaluated = false;

        renderScientificState();
    }


    private int countCharacter(
            String value,
            char target
    ) {

        if (value == null) {
            return 0;
        }

        int count = 0;

        for (int index = 0;
             index < value.length();
             index++) {

            if (value.charAt(index)
                    == target) {

                count++;
            }
        }

        return count;
    }


    // =========================================================
    // PERCENT
    // =========================================================

    private void applyScientificPercent() {

        ScientificEngine.ScientificResult
                currentResult =
                evaluateScientificCurrentInput();

        if (!currentResult.isSuccess()) {

            applyScientificResult(
                    currentResult,
                    false
            );

            return;
        }

        String percentExpression =
                "("
                        + currentResult.getFormattedValue()
                        + ") ÷ 100";

        ScientificEngine.ScientificResult result =
                scientificEngine
                        .evaluateExpression(
                                percentExpression
                        );

        boolean shouldSave =
                scientificPendingOperation
                        == ScientificPendingOperation.NONE;

        applyScientificResult(
                result,
                shouldSave
        );
    }


    // =========================================================
    // TOGGLE SIGN
    // =========================================================

    private void toggleScientificSign() {

        prepareScientificForValueInput();

        if (scientificInput == null
                || scientificInput.trim().isEmpty()) {

            scientificInput = "-";

        } else if (scientificInput.startsWith("-(")
                && scientificInput.endsWith(")")) {

            scientificInput =
                    scientificInput.substring(
                            2,
                            scientificInput.length() - 1
                    );

        } else {

            scientificInput =
                    "-(" + scientificInput + ")";
        }

        scientificJustEvaluated = false;

        if (scientificPendingOperation
                == ScientificPendingOperation.NONE) {

            scientificExpression = "";
        }

        renderScientificState();
    }


    // =========================================================
    // BACKSPACE
    // =========================================================

    private void scientificBackspace() {

        if (scientificJustEvaluated) {

            clear();

            return;
        }

        if (scientificInput == null
                || scientificInput.isEmpty()) {

            return;
        }

        scientificInput =
                scientificInput.substring(
                        0,
                        scientificInput.length() - 1
                ).trim();

        renderScientificState();
    }


    // =========================================================
    // INPUT PREPARATION
    // =========================================================

    private void prepareScientificForValueInput() {

        if (!scientificJustEvaluated) {
            return;
        }

        if (scientificPendingOperation
                != ScientificPendingOperation.NONE) {

            scientificInput = "";
            scientificResult = "0";
            scientificJustEvaluated = false;

            return;
        }

        scientificInput = "";
        scientificExpression = "";
        scientificResult = "0";
        scientificJustEvaluated = false;
    }


    // =========================================================
    // RESET
    // =========================================================

    private void resetScientificState() {

        scientificInput = "";
        scientificExpression = "";
        scientificResult = "0";

        scientificJustEvaluated = false;

        scientificPendingOperation =
                ScientificPendingOperation.NONE;

        scientificPendingFirstValue = 0.0;
        scientificPendingFirstText = "";
    }


    // =========================================================
    // RENDER
    // =========================================================

    private void renderScientificState() {

        if (tvSciExpression == null
                || tvSciResult == null
                || tvSciAngleIndicator == null
                || btnSciAngleMode == null) {

            return;
        }

        boolean isDegree =
                scientificEngine.getAngleMode()
                        == AngleMode.DEG;

        tvSciAngleIndicator.setText(
                isDegree
                        ? R.string.sci_deg
                        : R.string.sci_rad
        );

        btnSciAngleMode.setText(
                isDegree
                        ? R.string.sci_deg
                        : R.string.sci_rad
        );

        String displayExpression =
                scientificExpression == null
                        ? ""
                        : scientificExpression;

        if (scientificPendingOperation
                == ScientificPendingOperation.POWER) {

            displayExpression =
                    scientificPendingFirstText + " ^";

            if (scientificInput != null
                    && !scientificInput
                    .trim()
                    .isEmpty()) {

                displayExpression +=
                        " " + scientificInput;
            }

        } else if (scientificPendingOperation
                == ScientificPendingOperation.Y_ROOT) {

            displayExpression =
                    scientificPendingFirstText + "√";

            if (scientificInput != null
                    && !scientificInput
                    .trim()
                    .isEmpty()) {

                displayExpression +=
                        " " + scientificInput;
            }
        }

        tvSciExpression.setText(
                displayExpression
        );

        String displayResult;

        if (!scientificJustEvaluated
                && scientificInput != null
                && !scientificInput
                .trim()
                .isEmpty()) {

            displayResult = scientificInput;

        } else {

            displayResult = scientificResult;
        }

        if (displayResult == null
                || displayResult.trim().isEmpty()) {

            displayResult = "0";
        }

        tvSciResult.setText(
                displayResult
        );
    }
}
```

---

## Không sửa `MainActivity` trong R1A

Giữ nguyên toàn bộ code 8.7 đang chạy.

Chưa xóa các field:

```text
scientificEngine
scientificInput
scientificExpression
scientificResult
scientificJustEvaluated
scientificPendingOperation
scientificPendingFirstValue
scientificPendingFirstText
```

Chưa xóa các method:

```text
setupScientificDigitButtons()
bindScientificDigitButton()
appendScientificDigit()

appendScientificDecimal()
getScientificCurrentNumberToken()

setupScientificOperatorButtons()
bindScientificOperatorButton()
appendScientificOperator()
removeTrailingScientificOperator()

setupScientificFunctionButtons()

performScientificUnary()
evaluateScientificCurrentInput()

beginScientificPendingOperation()
clearScientificPendingOperation()
calculateScientificExpression()

applyScientificResult()
saveScientificHistory()

appendScientificConstant()
appendScientificOpenParenthesis()
appendScientificCloseParenthesis()
countCharacter()

applyScientificPercent()
toggleScientificSign()
scientificBackspace()
prepareScientificForValueInput()
resetScientificState()
renderScientificState()
```

Các field và method trên vẫn đang là implementation thật cho đến khi thực hiện R1B.

---

## Build checkpoint

Chạy:

```text
Build
→ Make Project
```

Checklist:

```text
[ ] Package ui.controller được tạo

[ ] ScientificCalculatorController.java compile

[ ] Không báo thiếu R.id.btnSci...

[ ] Không báo thiếu ScientificEngine method

[ ] Các method reference compile:
    scientificEngine::square
    scientificEngine::cube
    scientificEngine::sin
    scientificEngine::log10
    scientificEngine::squareRoot

[ ] MainActivity chưa thay đổi

[ ] App vẫn chạy đúng như checkpoint 8.7
```

## Bước tiếp theo

Sau khi Build pass:

```text
R1B
→ nối ScientificCalculatorController vào MainActivity
→ xóa Scientific fields/methods cũ
→ Build
→ Runtime test Scientific
```
