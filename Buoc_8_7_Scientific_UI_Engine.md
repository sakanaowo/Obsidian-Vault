# Bước 8.7 — Scientific UI ↔ Engine

Kết nối giao diện Scientific với `ScientificEngine`.

```text
Scientific button
        ↓
MainActivity
        ↓
ScientificEngine
        ↓
ScientificResult
        ↓
tvSciExpression / tvSciResult
        ↓
HistoryManager
```

## 8.7.1 — Thêm imports

Trong `MainActivity.java`:

```java
import com.example.calculator.engine.AngleMode;
import com.example.calculator.engine.ScientificEngine;
```

## 8.7.2 — Thêm fields

```java
private ScientificEngine scientificEngine;

private TextView tvSciExpression;
private TextView tvSciResult;
private TextView tvSciAngleIndicator;

private String scientificInput;
private String scientificExpression;
private String scientificResult;

private boolean scientificJustEvaluated;

private ScientificPendingOperation scientificPendingOperation;
private double scientificPendingFirstValue;
private String scientificPendingFirstText;

private enum ScientificPendingOperation {
    NONE,
    POWER,
    Y_ROOT
}

private interface ScientificUnaryOperation {
    ScientificEngine.ScientificResult calculate(double value);
}
```

## 8.7.3 — Khởi tạo

Trong `onCreate()`:

```java
calculatorEngine = new CalculatorEngine();

scientificEngine = new ScientificEngine();

historyManager = new HistoryManager(
        getApplicationContext()
);

currentMode = CalculatorMode.BASIC;

resetScientificState();
```

## 8.7.4 — Bind Scientific views

Bổ sung vào `bindViews()`:

```java
tvSciExpression =
        scientificCalculatorLayout.findViewById(
                R.id.tvSciExpression
        );

tvSciResult =
        scientificCalculatorLayout.findViewById(
                R.id.tvSciResult
        );

tvSciAngleIndicator =
        scientificCalculatorLayout.findViewById(
                R.id.tvSciAngleIndicator
        );
```

Helper:

```java
private View findScientificView(int viewId) {
    return scientificCalculatorLayout.findViewById(viewId);
}
```

## 8.7.5 — Gọi setup trong `onCreate()`

```java
setupDigitButtons();
setupOperatorButtons();
setupFunctionButtons();

setupScientificDigitButtons();
setupScientificOperatorButtons();
setupScientificFunctionButtons();

setupHistoryButton();
setupModeButton();

renderCalculatorMode();
renderState();
renderScientificState();
```

## 8.7.6 — Scientific digit buttons

```java
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

    findScientificView(buttonId)
            .setOnClickListener(view ->
                    appendScientificDigit(digit)
            );
}

private void appendScientificDigit(String digit) {

    if (digit == null || !digit.matches("[0-9]")) {
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
```

## 8.7.7 — Decimal

```java
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

    int index = scientificInput.length() - 1;

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
```

## 8.7.8 — Arithmetic operators

```java
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

    findScientificView(buttonId)
            .setOnClickListener(view ->
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

    scientificInput += " " + operator + " ";

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
```

## 8.7.9 — Setup Scientific function buttons

```java
private void setupScientificFunctionButtons() {

    findScientificView(R.id.btnSciDecimal)
            .setOnClickListener(view ->
                    appendScientificDecimal()
            );

    findScientificView(R.id.btnSciEquals)
            .setOnClickListener(view ->
                    calculateScientificExpression()
            );

    findScientificView(R.id.btnSciClear)
            .setOnClickListener(view -> {
                resetScientificState();
                renderScientificState();
            });

    findScientificView(R.id.btnSciBackspace)
            .setOnClickListener(view ->
                    scientificBackspace()
            );

    findScientificView(R.id.btnSciToggleSign)
            .setOnClickListener(view ->
                    toggleScientificSign()
            );

    findScientificView(R.id.btnSciPercent)
            .setOnClickListener(view ->
                    applyScientificPercent()
            );

    findScientificView(R.id.btnSciOpenParen)
            .setOnClickListener(view ->
                    appendScientificOpenParenthesis()
            );

    findScientificView(R.id.btnSciCloseParen)
            .setOnClickListener(view ->
                    appendScientificCloseParenthesis()
            );

    findScientificView(R.id.btnSciSquare)
            .setOnClickListener(view ->
                    performScientificUnary(
                            scientificEngine::square
                    )
            );

    findScientificView(R.id.btnSciCube)
            .setOnClickListener(view ->
                    performScientificUnary(
                            scientificEngine::cube
                    )
            );

    findScientificView(R.id.btnSciPower)
            .setOnClickListener(view ->
                    beginScientificPendingOperation(
                            ScientificPendingOperation.POWER
                    )
            );

    findScientificView(R.id.btnSciReciprocal)
            .setOnClickListener(view ->
                    performScientificUnary(
                            scientificEngine::reciprocal
                    )
            );

    findScientificView(R.id.btnSciSqrt)
            .setOnClickListener(view ->
                    performScientificUnary(
                            scientificEngine::squareRoot
                    )
            );

    findScientificView(R.id.btnSciCbrt)
            .setOnClickListener(view ->
                    performScientificUnary(
                            scientificEngine::cubeRoot
                    )
            );

    findScientificView(R.id.btnSciYRoot)
            .setOnClickListener(view ->
                    beginScientificPendingOperation(
                            ScientificPendingOperation.Y_ROOT
                    )
            );

    findScientificView(R.id.btnSciFactorial)
            .setOnClickListener(view ->
                    performScientificUnary(
                            scientificEngine::factorial
                    )
            );

    findScientificView(R.id.btnSciSin)
            .setOnClickListener(view ->
                    performScientificUnary(
                            scientificEngine::sin
                    )
            );

    findScientificView(R.id.btnSciCos)
            .setOnClickListener(view ->
                    performScientificUnary(
                            scientificEngine::cos
                    )
            );

    findScientificView(R.id.btnSciTan)
            .setOnClickListener(view ->
                    performScientificUnary(
                            scientificEngine::tan
                    )
            );

    findScientificView(R.id.btnSciLn)
            .setOnClickListener(view ->
                    performScientificUnary(
                            scientificEngine::naturalLog
                    )
            );

    findScientificView(R.id.btnSciLog10)
            .setOnClickListener(view ->
                    performScientificUnary(
                            scientificEngine::log10
                    )
            );

    findScientificView(R.id.btnSciE)
            .setOnClickListener(view ->
                    appendScientificConstant("e")
            );

    findScientificView(R.id.btnSciPi)
            .setOnClickListener(view ->
                    appendScientificConstant("π")
            );

    findScientificView(R.id.btnSciExp)
            .setOnClickListener(view ->
                    performScientificUnary(
                            scientificEngine::exponential
                    )
            );

    findScientificView(R.id.btnSciTenPower)
            .setOnClickListener(view ->
                    performScientificUnary(
                            scientificEngine::tenPower
                    )
            );

    findScientificView(R.id.btnSciAngleMode)
            .setOnClickListener(view -> {
                scientificEngine.toggleAngleMode();
                renderScientificState();
            });
}
```

## 8.7.10 — Unary operation

```java
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

    return scientificEngine.evaluateExpression(source);
}
```

## 8.7.11 — `xʸ` và `ʸ√x`

```java
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
```

## 8.7.12 — Scientific Equals

```java
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
                scientificEngine.evaluateExpression(
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
```

## 8.7.13 — Apply result và lưu History

```java
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
                && !result.getErrorMessage()
                .trim().isEmpty()) {

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

    if (result == null
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
```

## 8.7.14 — Constants `π` và `e`

```java
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
```

## 8.7.15 — Parentheses

```java
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

        if (value.charAt(index) == target) {
            count++;
        }
    }

    return count;
}
```

## 8.7.16 — Percent

```java
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
            scientificEngine.evaluateExpression(
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
```

## 8.7.17 — Toggle sign

```java
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
```

## 8.7.18 — Backspace

```java
private void scientificBackspace() {

    if (scientificJustEvaluated) {

        resetScientificState();
        renderScientificState();

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
```

## 8.7.19 — Chuẩn bị input mới

```java
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
```

## 8.7.20 — Reset Scientific state

```java
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
```

`AC` không reset `AngleMode`; `DEG/RAD` được giữ nguyên.

## 8.7.21 — Render Scientific state

```java
private void renderScientificState() {

    if (scientificEngine == null
            || tvSciExpression == null
            || tvSciResult == null
            || tvSciAngleIndicator == null) {

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

    Button angleButton =
            scientificCalculatorLayout.findViewById(
                    R.id.btnSciAngleMode
            );

    angleButton.setText(
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
                && !scientificInput.trim().isEmpty()) {

            displayExpression +=
                    " " + scientificInput;
        }

    } else if (scientificPendingOperation
            == ScientificPendingOperation.Y_ROOT) {

        displayExpression =
                scientificPendingFirstText + "√";

        if (scientificInput != null
                && !scientificInput.trim().isEmpty()) {

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
            && !scientificInput.trim().isEmpty()) {

        displayResult = scientificInput;

    } else {

        displayResult = scientificResult;
    }

    if (displayResult == null
            || displayResult.trim().isEmpty()) {

        displayResult = "0";
    }

    tvSciResult.setText(displayResult);
}
```

---

# Build checkpoint

```text
Build
→ Make Project
```

Checklist:

```text
[ ] ScientificEngine import resolve
[ ] AngleMode import resolve
[ ] tvSciExpression resolve
[ ] tvSciResult resolve
[ ] tvSciAngleIndicator resolve
[ ] btnSci0–btnSci9 resolve
[ ] btnSciAdd/Subtract/Multiply/Divide resolve
[ ] btnSciEquals resolve
[ ] ScientificUnaryOperation compile
[ ] ScientificPendingOperation compile
[ ] Method references compile
[ ] MainActivity compile
```

# Runtime tests

## Numeric và arithmetic

```text
2 + 3 =          → 5
2 + 3 × 4 =      → 14
(2 + 3) × 4 =    → 20
```

## Unary Scientific

```text
16 √x     → 4
2 x³      → 8
4 1/x     → 0.25
5 x!      → 120
```

## Power và Y Root

```text
2 xʸ 3 =       → 8
3 ʸ√x 8 =      → 2
2 ʸ√x -4 =     → Error
```

## DEG/RAD

```text
DEG: sin(30)              → 0.5
RAD: sin(1.570796326795)  → 1
```

## Logs và errors

```text
log(100)   → 2
ln(0)      → Error
1/0        → Error
tan(90°)   → Error
```

## Constants

```text
π          → 3.14159265359...
2 × π      → 6.28318530718...
e¹         → 2.718281828459...
10³        → 1000
```

## Regression

```text
Basic: 25 × 4 = 100
Scientific History hoạt động
Mode Basic / Scientific / Convert không crash
```

# Trạng thái sau Bước 8.7

```text
M3 — SCIENTIFIC CALCULATOR

[✓] 8.5 Scientific UI

[~] 8.6 ScientificEngine
    Chờ xác nhận Build/Test thực tế

[~] 8.7 Scientific UI ↔ Engine
    ├── numeric keypad
    ├── decimal
    ├── operators
    ├── parentheses
    ├── x² / x³
    ├── xʸ
    ├── √x / ∛x / ʸ√x
    ├── 1/x
    ├── x!
    ├── sin/cos/tan
    ├── ln/log
    ├── π/e
    ├── eˣ/10ˣ
    ├── Deg/Rad
    ├── error rendering
    └── Scientific History

NEXT

8.8 Scientific Functional Test
    ├── expression precedence
    ├── chained calculations
    ├── domain errors
    ├── long-number formatting
    ├── mode switching preservation
    └── History duplicate prevention
```
