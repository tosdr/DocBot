import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*private messages)|(?=.*review)((?=.*message)|(?=.*messaging)|(?=.*private communication))|((?=.*monitor)|(?=.*record)|(?=.*process))((?=.*communication)|(?=.*message)|(?=.*messaging))|(?=.*private)(?=.*chat)(?=.*moderation))", "i"),
	expressionDont: new RegExp("", "i"),
	caseID: 239,
	name: "The service can read your private messages"
} as Regex;